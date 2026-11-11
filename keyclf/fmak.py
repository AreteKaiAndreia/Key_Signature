"""Plan and resume a disk-bounded, artist-separated FMAKv2 pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import tempfile
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from itertools import islice
from pathlib import Path, PurePosixPath
from threading import BoundedSemaphore, local

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import StratifiedGroupKFold

from .features import config_signature, extract_features, feature_names
from .keys import key_name, parse_key
from .rangezip import MAX_MEMBER, open_member, open_zip, read_member

ROOT = Path("data/work/fmak-pilot")
GSK = Path("data/annotations/giantsteps-key-dataset")
PROTOCOL = "fmak-v2-artist-balanced-cqt-1"
SEED = 42
RECORD = "https://zenodo.org/records/10719860/files/"
FMA_METADATA = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
LABEL_SOURCES = {
    "fmakv2.csv": ("https://zenodo.org/records/12759100/files/fmakv2.csv?download=1", "3b2d16784ffbda850c8ddf0519478bfd"),
    "fmak.csv": (RECORD + "fma_keys_metadata.csv?download=1", "d80a03bc8659edc60e335bd7f6bdf12a"),
    "archives.json": ("https://zenodo.org/api/records/10719860", None),
}
HISTOGRAM = feature_names()[24:]


def atomic_bytes(path: Path, data: bytes) -> None:
    """Write a checkpoint beside its destination and replace it atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".pending-", delete=False) as file:
        pending = Path(file.name)
        try:
            file.write(data)
            file.close()
            os.replace(pending, path)
        finally:
            pending.unlink(missing_ok=True)


def atomic_json(path: Path, value: dict) -> None:
    """Write a recoverable JSON checkpoint."""
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def digest(value: dict) -> str:
    """Hash structured inputs independently of JSON formatting."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_inputs(session, work: Path) -> dict[str, bytes]:
    """Acquire the small label files and check publisher-supplied checksums."""
    inputs = {}
    for name, (url, checksum) in LABEL_SOURCES.items():
        path = work / "inputs" / name
        if path.exists():
            content = path.read_bytes()
        else:
            with session.get(url, stream=True, timeout=(12, 30)) as response:
                response.raise_for_status()
                content = response.raw.read(1024**2 + 1)
            if not 0 < len(content) <= 1024**2:
                raise ValueError(f"unexpected metadata size: {name}")
        if checksum and hashlib.md5(content).hexdigest() != checksum:
            raise ValueError(f"published checksum mismatch: {name}")
        if not path.exists():
            atomic_bytes(path, content)
        inputs[name] = content
    return inputs


def read_labels(content: bytes) -> dict[int, int | None]:
    """Read one key per unique track ID; mark non-major/minor labels ineligible."""
    labels = {}
    for row in csv.DictReader(io.StringIO(content.decode("utf-8-sig"))):
        track = int(row["track_id"])
        if track in labels:
            raise ValueError(f"duplicate FMAK label ID {track}")
        label = row["key_and_mode"].strip().lower()
        try:
            labels[track] = parse_key(label) if label.endswith((" major", " minor")) else None
        except ValueError:
            labels[track] = None
    return labels


def allowed_license(value: str) -> bool:
    """Allow named CC attribution licenses without NoDerivatives for local research."""
    text = value.casefold().strip().replace("non-commercial", "noncommercial").replace("share-alike", "sharealike")
    if text.startswith("cc0 1.0") or text == "public domain":
        return True
    return bool(re.fullmatch(
        r"attribution(?:-noncommercial)?(?:-sharealike)? \d+\.\d+(?: (?:international|unported|united states|generic))?",
        text,
    ))


def get_fma_metadata(session, work: Path, wanted: set[int]) -> dict:
    """Stream only the track-table member, retaining metadata for relevant IDs."""
    target = work / "inputs" / "track_metadata.json"
    if target.exists():
        cached = json.loads(target.read_text())
        if cached.get("requested_ids") == sorted(wanted):
            return cached
    stream, archive = open_zip(FMA_METADATA, session)
    with archive:
        info = archive.getinfo("fma_metadata/tracks.csv")
        rows = {}
        csv.field_size_limit(8 * 1024**2)
        with open_member(stream, archive, info, max_size=300 * 1024**2) as member:
            reader = csv.reader(io.TextIOWrapper(member, encoding="utf-8-sig", newline=""))
            columns = list(zip(next(reader), next(reader), strict=True))
            artist, license_col, duration = (columns.index(column) for column in
                                               (("artist", "id"), ("track", "license"), ("track", "duration")))
            for row in reader:
                if not row or not row[0].isdigit() or int(row[0]) not in wanted:
                    continue
                if row[artist].isdigit() and row[duration].isdigit():
                    rows[row[0]] = {"artist_id": int(row[artist]), "license": row[license_col],
                                    "duration_s": int(row[duration])}
        result = {"requested_ids": sorted(wanted), "tracks": rows, "source": FMA_METADATA,
                  "member_crc32": f"{info.CRC:08x}"}
    atomic_json(target, result)
    return result


def archive_catalog(content: bytes) -> dict:
    """Get published shard sizes and checksums without trusting arbitrary URLs."""
    record = json.loads(content)
    if int(record["id"]) != 10719860:
        raise ValueError("unexpected FMAK audio record")
    result = {item["key"]: {"size": int(item["size"]), "checksum": item["checksum"]}
              for item in record["files"] if re.fullmatch(r"\d{3}-\d{3}\.zip", item["key"])}
    if len(result) != 10:
        raise ValueError("unexpected FMAK shard catalog")
    return result


def archive_url(shard: str) -> str:
    """Construct the URL of a member of the pinned Zenodo release."""
    if not re.fullmatch(r"\d{3}-\d{3}\.zip", shard):
        raise ValueError("invalid FMAK shard name")
    return RECORD + shard + "?download=1"


def index_audio(session, archives: dict, work: Path) -> dict[str, dict]:
    """Read only ZIP directories and index MP3 metadata, not audio bytes."""
    path = work / "inputs" / "audio_index.json"
    prior = json.loads(path.read_text()) if path.exists() else {}
    if prior.get("archives") != archives:
        prior = {"archives": archives, "shards": {}}
    for shard, spec in sorted(archives.items()):
        if shard in prior["shards"]:
            continue
        _, archive = open_zip(archive_url(shard), session, spec["size"])
        indexed = {}
        with archive:
            for info in archive.infolist():
                match = re.fullmatch(r"(\d{6})\.mp3", PurePosixPath(info.filename).name, re.I)
                if not match or not 0 < info.file_size <= MAX_MEMBER or not 0 < info.compress_size <= MAX_MEMBER:
                    continue
                track = str(int(match[1]))
                if track in indexed:
                    raise ValueError(f"ambiguous FMAK archive member {track}")
                indexed[track] = {"archive": shard, "member": info.filename,
                                  "crc32": f"{info.CRC:08x}", "audio_bytes": info.file_size}
        prior["shards"][shard] = indexed
        atomic_json(path, prior)
        print(f"indexed {shard}: {len(indexed)} audio members", flush=True)
    result = {}
    for indexed in prior["shards"].values():
        if set(result) & set(indexed):
            raise ValueError("same recording ID in multiple FMAK shards")
        result.update(indexed)
    return result


def balanced_pick(rows: list[dict], count: int, seed: int) -> list[dict]:
    """Draw about 60% major; cycle through available keys within each mode."""
    rng = np.random.default_rng(seed)
    queues = {k: [row for row in rows if row["key_id"] == k] for k in range(24)}
    for queue in queues.values():
        rng.shuffle(queue)
    chosen = []
    for keys, amount in ((range(12), round(count * .6)), (range(12, 24), count - round(count * .6)), (range(24), count)):
        stop = min(count, len(chosen) + amount)
        while len(chosen) < stop and any(queues[k] for k in keys):
            for k in rng.permutation(list(keys)):
                if queues[k]:
                    chosen.append(queues[k].pop())
                if len(chosen) == stop:
                    break
    if len(chosen) != count:
        raise ValueError(f"split has only {len(chosen)} eligible recordings; need {count}")
    return chosen


def select_pilot(rows: list[dict], count: int) -> list[dict]:
    """Assign artists before selecting key-balanced recordings inside each split."""
    rows = sorted(rows, key=lambda row: row["track_id"])
    splitter = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=SEED)
    partitions = {"train": [], "val": [], "test": []}
    for fold, (_, indices) in enumerate(splitter.split(rows, [r["key_id"] for r in rows], [r["artist_id"] for r in rows])):
        name = "test" if fold < 3 else "val" if fold < 6 else "train"
        partitions[name].extend(rows[i] for i in indices)
    heldout = round(.15 * count)
    sizes = {"train": count - 2 * heldout, "val": heldout, "test": heldout}
    selected = [dict(row, split=name) for i, (name, part) in enumerate(partitions.items())
                for row in balanced_pick(part, sizes[name], SEED + i)]
    return sorted(selected, key=lambda row: (row["archive"], row["track_id"]))


def load_manifest(work: Path) -> dict:
    """Validate the frozen sample and check that no artist crosses splits."""
    manifest = json.loads((work / "manifest.json").read_text())
    signature = manifest.pop("signature")
    if manifest["protocol"] != PROTOCOL or digest(manifest) != signature:
        raise ValueError("incompatible or modified FMAK manifest")
    rows = manifest["tracks"]
    if len({r["track_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate FMAK recording IDs")
    artists: dict[int, str] = {}
    for row in rows:
        if artists.setdefault(row["artist_id"], row["split"]) != row["split"]:
            raise ValueError("FMAK artist crosses partitions")
    return dict(manifest, signature=signature)


def plan(work: Path, count: int) -> dict:
    """Freeze labels, eligible audio, artist groups, and the sample before extraction."""
    if (work / "manifest.json").exists():
        result = load_manifest(work)
        if len(result["tracks"]) != count:
            raise ValueError("use a separate work directory to change a frozen pilot size")
        return result
    with requests.Session() as session:
        inputs = read_inputs(session, work)
        labels, original = read_labels(inputs["fmakv2.csv"]), read_labels(inputs["fmak.csv"])
        agreed = {i: y for i, y in labels.items() if y is not None and original.get(i) == y}
        metadata = get_fma_metadata(session, work, set(agreed))
        archives = archive_catalog(inputs["archives.json"])
        members = index_audio(session, archives, work)
    eligible, hashes = [], set()
    for track, label in sorted(agreed.items()):
        audio = members.get(str(track))
        info = metadata["tracks"].get(str(track))
        if not audio or not info or not 10 <= info["duration_s"] <= 600 or not allowed_license(info["license"]):
            continue
        signature = audio["crc32"], audio["audio_bytes"]
        if signature in hashes:
            continue
        hashes.add(signature)
        eligible.append({"track_id": f"fmak-{track:06d}", "key_id": label, "key": key_name(label),
                         **audio, **info})
    print(f"labels={len(labels)} agreed={len(agreed)} eligible={len(eligible)}", flush=True)
    selected = select_pilot(eligible, count)
    result = {"protocol": PROTOCOL, "seed": SEED, "features": config_signature(),
              "archives": archives, "tracks": selected, "label_rows": len(labels),
              "agreed_labels": len(agreed), "eligible_rows": len(eligible),
              "metadata_crc32": metadata["member_crc32"]}
    atomic_json(work / "manifest.json", dict(result, signature=digest(result)))
    return load_manifest(work)


def original_checksums(annotations: Path) -> set[str]:
    """Require a complete original checksum catalog before accepting added audio."""
    hashes = {file.read_text().split()[0].lower() for file in (annotations / "md5").glob("*.md5")}
    if len(hashes) != 604 or not all(re.fullmatch(r"[0-9a-f]{32}", h) for h in hashes):
        raise ValueError("expected 604 distinct GiantSteps audio checksums")
    return hashes


def validate_checkpoint(row: dict, track: dict, signature: str) -> bool:
    """Accept only a finite normalized histogram for this frozen recording."""
    expected = ("track_id", "key", "split", "artist_id", "crc32")
    if row.get("manifest") != signature or any(row.get(k) != track[k] for k in expected):
        return False
    if not re.fullmatch(r"[0-9a-f]{32}", str(row.get("md5", ""))):
        return False
    try:
        values = np.array([row[name] for name in HISTOGRAM], dtype=float)
        return bool(np.isfinite(values).all() and (values >= 0).all() and np.isclose(values.sum(), 1))
    except (KeyError, ValueError, TypeError):
        return False


def extract(work: Path, annotations: Path, workers: int, limit: int | None = None) -> int:
    """Fetch selected members only, save atomic checkpoints, and remove audio."""
    manifest = load_manifest(work)
    all_tracks, signature = manifest["tracks"], manifest["signature"]
    if manifest["features"] != config_signature():
        raise ValueError("feature configuration differs from frozen manifest")
    old_hashes = original_checksums(annotations)
    tracks = all_tracks if limit is None else all_tracks[:limit]
    thread = local()
    decoding = BoundedSemaphore(2)

    def one(track: dict) -> tuple[str, dict]:
        target = work / "checkpoints" / f"{track['track_id']}.json"
        if target.exists():
            prior = json.loads(target.read_text())
            if validate_checkpoint(prior, track, signature) and prior.get("md5") not in old_hashes:
                return "resumed", prior
        if not hasattr(thread, "session"):
            thread.session = requests.Session()
            thread.shard = None
        started = time.perf_counter()
        shard = track["archive"]
        if thread.shard != shard:
            if thread.shard is not None:
                thread.archive.close()
                thread.stream.close()
            thread.stream, thread.archive = open_zip(archive_url(shard), thread.session, manifest["archives"][shard]["size"])
            thread.shard = shard
        info = thread.archive.getinfo(track["member"])
        if f"{info.CRC:08x}" != track["crc32"] or info.file_size != track["audio_bytes"]:
            raise ValueError("remote member changed after planning")
        audio = read_member(thread.stream, thread.archive, info)
        download_s = time.perf_counter() - started
        md5 = hashlib.md5(audio).hexdigest()
        if md5 in old_hashes:
            raise ValueError("audio duplicates an original GiantSteps preview")
        with decoding, tempfile.NamedTemporaryFile(suffix=".mp3", prefix="keyclf-fmak-") as temporary:
            started = time.perf_counter()
            temporary.write(audio)
            temporary.flush()
            histogram = extract_features(temporary.name)[24:]
        row = dict(track, manifest=signature, md5=md5, download_s=download_s,
                   extraction_s=time.perf_counter() - started,
                   **dict(zip(HISTOGRAM, map(float, histogram), strict=True)))
        if not validate_checkpoint(row, track, signature):
            raise ValueError("invalid extracted histogram")
        atomic_json(target, row)
        return "extracted", row

    counters, results, failures = Counter(), [], {}
    failure_streak = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = iter(tracks)
        active = {pool.submit(one, track): track for track in islice(pending, workers)}
        while active:
            completed, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in completed:
                track = active.pop(future)
                try:
                    status, row = future.result()
                    counters[status] += 1
                    results.append(row)
                    if status == "extracted":
                        failure_streak = 0
                except Exception as exc:
                    failure_streak += 1
                    failures[track["track_id"]] = str(exc)
                    print(f"failed {track['track_id']}: {exc}", flush=True)
                done = len(results) + len(failures)
                if done % 10 == 0 or done == len(tracks):
                    print(f"{done}/{len(tracks)} checked, {len(results)} ready, {len(failures)} failed", flush=True)
                # Drain active work instead of making hundreds of requests
                # when the first batch reveals a systemic source failure.
                if failure_streak < 6:
                    upcoming = next(pending, None)
                    if upcoming is not None:
                        active[pool.submit(one, upcoming)] = upcoming
    complete = not failures and len(results) == len(all_tracks)
    if complete:
        if len({row["md5"] for row in results}) != len(results):
            raise ValueError("duplicate audio inside FMAK pilot")
        frame = pd.DataFrame(results).sort_values("track_id").reset_index(drop=True).sort_index(axis=1)
        cache = work / "features.parquet"
        if not cache.exists() or not pd.read_parquet(cache).equals(frame):
            with tempfile.NamedTemporaryFile(dir=work, suffix=".parquet") as pending:
                frame.to_parquet(pending.name, index=False)
                atomic_bytes(cache, Path(pending.name).read_bytes())
    result = {"completed": len(results), "planned": len(all_tracks), "complete": complete,
              "counts": dict(counters), "failures": failures, "raw_audio_retained": False}
    atomic_json(work / "extraction.json", result)
    print(json.dumps(result), flush=True)
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    """Build the pilot manifest or resume its feature extraction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("plan", "extract"))
    parser.add_argument("--work", type=Path, default=ROOT)
    parser.add_argument("--annotations", type=Path, default=GSK)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=8,
                        help="parallel downloads (audio decoding is limited to two)")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.count < 24 or (args.limit is not None and args.limit < 1):
        parser.error("count must be at least 24 and limit must be positive")
    if args.operation == "plan":
        result = plan(args.work, args.count)
        print(json.dumps({"manifest": result["signature"], "count": len(result["tracks"]),
                          "splits": Counter(row["split"] for row in result["tracks"])}))
        return 0
    return extract(args.work, args.annotations, args.workers, args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
