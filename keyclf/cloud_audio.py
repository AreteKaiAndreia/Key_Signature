# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded, temporary decoding for Streamlit uploads and recordings."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .audio_settings import MAX_BYTES, MAX_SECONDS, SAMPLE_RATE
FORMATS = {".wav": "wav", ".mp3": "mp3", ".mp4": "mov"}


def ffmpeg_executable() -> str:
    """Use the pinned wheel's FFmpeg without a system multimedia installation."""
    from imageio_ffmpeg import get_ffmpeg_exe

    return get_ffmpeg_exe()


def decode_clip(content: bytes, filename: str) -> tuple[np.ndarray, bool]:
    """Decode up to 120 seconds plus one sample to detect truncation."""
    suffix = Path(filename).suffix.lower()
    if suffix not in FORMATS:
        raise ValueError("Choose an MP3 or MP4 file, or record a soundbite.")
    if not 0 < len(content) <= MAX_BYTES:
        raise ValueError("The audio must be nonempty and no larger than 50 MB.")
    max_samples = MAX_SECONDS * SAMPLE_RATE
    # Let FFmpeg decode up to one extra sample after resampling. The explicit
    # demuxer, path, and protocol allowlist avoid treating an upload as a URL.
    with tempfile.NamedTemporaryFile(suffix=suffix, prefix="keyclip-") as upload:
        upload.write(content)
        upload.flush()
        try:
            decoded = subprocess.run(
                [ffmpeg_executable(), "-nostdin", "-hide_banner", "-loglevel", "error",
                 "-protocol_whitelist", "file,pipe", "-f", FORMATS[suffix],
                 "-threads", "1", "-i", upload.name, "-map", "0:a:0", "-vn", "-sn", "-dn",
                 "-af", f"aresample={SAMPLE_RATE},atrim=end_sample={max_samples + 1}",
                 "-t", f"{MAX_SECONDS + 1/SAMPLE_RATE:.9f}", "-ac", "1", "-ar", str(SAMPLE_RATE),
                 "-c:a", "pcm_f32le", "-f", "f32le", "pipe:1"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=20, check=True,
            )
        except subprocess.TimeoutExpired as exc:
            raise ValueError("Decoding timed out. Try a shorter soundbite.") from exc
        except subprocess.CalledProcessError as exc:
            raise ValueError("No readable audio stream was found in the file.") from exc
        except FileNotFoundError as exc:
            raise RuntimeError("FFmpeg is not installed.") from exc
    if len(decoded.stdout) % 4 or len(decoded.stdout) > 4 * (max_samples + 1):
        raise ValueError("Invalid decoded audio.")
    samples = np.frombuffer(decoded.stdout, dtype="<f4")
    clipped = samples.size > max_samples
    samples = samples[:max_samples]
    if samples.size < SAMPLE_RATE // 2 or not np.isfinite(samples).all():
        raise ValueError("Use a soundbite with at least half a second of valid audio.")
    if float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)))) < 1e-5:
        raise ValueError("The clip is silent or too quiet to identify a key.")
    return samples.copy(), clipped
