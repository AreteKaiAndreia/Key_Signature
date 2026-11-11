"""Bounded reads of selected members from a ZIP served over HTTPS."""

from __future__ import annotations

import io
import re
import time
import zipfile

import requests
from urllib3.exceptions import HTTPError as TransportError

MAX_MEMBER = 64 * 1024**2
MAX_HEADER = 30 + 2 * 65535
MAX_RANGE = MAX_MEMBER + MAX_HEADER
_CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")


def fetch_range(session, url: str, start: int, count: int, size: int | None = None) -> tuple[bytes, int]:
    """Fetch exactly the requested bytes; never consume an unexpected full archive."""
    if start < 0 or not 0 < count <= MAX_RANGE:
        raise ValueError("invalid or oversized archive byte range")
    end = start + count - 1
    headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity", "Cache-Control": "no-cache"}
    for attempt in range(4):
        delay = 2**attempt
        try:
            with session.get(url, headers=headers, stream=True, timeout=(12, 45)) as response:
                if response.status_code in (429, 500, 502, 503, 504):
                    retry_after = response.headers.get("Retry-After", "")
                    if retry_after.isdigit():
                        delay = max(delay, min(60, int(retry_after)))
                    response.raise_for_status()
                bounds = _CONTENT_RANGE.fullmatch(response.headers.get("Content-Range", ""))
                if response.status_code != 206 or bounds is None:
                    raise ValueError("server did not honor the ZIP byte range; no full archive downloaded")
                lo, hi, total = map(int, bounds.groups())
                if (lo, hi) != (start, end) or hi >= total or (size is not None and total != size):
                    raise ValueError("ZIP range response has mismatched bounds or archive size")
                if response.headers.get("Content-Encoding", "identity") != "identity":
                    raise ValueError("encoded ZIP range response is not supported")
                content = response.raw.read(count + 1)
                if len(content) != count:
                    raise OSError("short or oversized ZIP range response")
                return content, total
        except (requests.RequestException, TransportError, OSError) as exc:
            if isinstance(exc, requests.HTTPError) and exc.response.status_code not in (429, 500, 502, 503, 504):
                raise
            if attempt == 3:
                raise
            time.sleep(delay)
    raise AssertionError("unreachable")


class RangeFile(io.RawIOBase):
    """Seekable, single-window view used by the standard library ZIP reader."""

    def __init__(self, url: str, size: int, session):
        super().__init__()
        self.url, self.size, self.session = url, size, session
        self.pos, self.start, self.buffer = 0, 0, b""

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        bases = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}
        if whence not in bases or bases[whence] + offset < 0:
            raise OSError("invalid ZIP seek")
        self.pos = bases[whence] + offset
        return self.pos

    def prefetch(self, start: int, count: int) -> None:
        count = min(count, self.size - start)
        if count > 0 and not (self.start <= start and start + count <= self.start + len(self.buffer)):
            self.buffer, _ = fetch_range(self.session, self.url, start, count, self.size)
            self.start = start

    def read(self, size: int = -1) -> bytes:
        size = self.size - self.pos if size < 0 else min(size, self.size - self.pos)
        if size <= 0:
            return b""
        self.prefetch(self.pos, size)
        offset = self.pos - self.start
        self.pos += size
        return self.buffer[offset:offset + size]


def open_zip(url: str, session, size: int | None = None) -> tuple[RangeFile, zipfile.ZipFile]:
    """Read the ZIP directory, including ZIP64 archives, without staging audio."""
    if size is None:
        _, size = fetch_range(session, url, 0, 1)
    stream = RangeFile(url, size, session)
    stream.prefetch(max(0, size - 256 * 1024), min(size, 256 * 1024))
    return stream, zipfile.ZipFile(stream)


def read_member(stream: RangeFile, archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    """Return one size-bounded member; ZipFile checks its CRC on complete read."""
    with open_member(stream, archive, info) as member:
        return member.read()


def open_member(stream: RangeFile, archive: zipfile.ZipFile, info: zipfile.ZipInfo, *, max_size: int = MAX_MEMBER):
    """Open one bounded compressed member; large metadata can be streamed."""
    if not 0 < info.file_size <= max_size or not 0 < info.compress_size <= MAX_MEMBER:
        raise ValueError("ZIP member exceeds the extraction size budget")
    stream.prefetch(info.header_offset, info.compress_size + MAX_HEADER)
    return archive.open(info)
