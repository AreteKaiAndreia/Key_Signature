"""Extract chroma from permissively licensed Freesound Loop Dataset audio."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import re
from threading import local
import zipfile

import numpy as np
import pandas as pd
import requests
import soundfile as sf
from threadpoolctl import threadpool_limits

from .features import DEFAULT_CONFIG, config_signature, feature_names, histogram_from_audio
from .fmak import atomic_bytes, atomic_json
from .keys import key_name, parse_key
from .rangezip import open_zip, read_member

ROOT = Path("data/work/open-key")
ARCHIVE = "https://zenodo.org/records/3967852/files/FSL10K.zip"
ANNOTATIONS = "https://zenodo.org/records/3967852/files/annotations.zip"
ANNOTATIONS_MD5 = "3920ee437802cf047a990b2968fa066c"
SIZE = 8841802341
LICENSES = {
    "http://creativecommons.org/publicdomain/zero/1.0/": "CC0-1.0",
    "http://creativecommons.org/licenses/by/3.0/": "CC-BY-3.0",
}
HISTOGRAM = feature_names()[24:]


def eligible(metadata: dict, annotations: zipfile.ZipFile) -> list[dict]:
    """Keep unanimous human key annotations and explicit CC0 or CC BY audio."""
    ratings = defaultdict(list)
    for name in annotations.namelist():
        match = re.fullmatch(r"annotations/\d+/sound-(\d+)\.json", name)
        if match:
            ratings[match[1]].append(json.loads(annotations.read(name)))
    rows = []
    for sound_id, labels in sorted(ratings.items(), key=lambda item: int(item[0])):
        sound = metadata.get(sound_id, {})
        license_url = sound.get("license")
        votes = {(row["key"], row["mode"]) for row in labels}
        if license_url not in LICENSES or len(votes) != 1 or any(row["discard"] for row in labels):
            continue
        pitch, mode = votes.pop()
        if mode not in {"maj", "min"} or pitch in {"none", "unknown"}:
            continue
        key = parse_key(f"{pitch} {'major' if mode == 'maj' else 'minor'}")
        rows.append({"id": int(sound_id), "creator": sound["username"], "title": sound["name"],
                     "license": LICENSES[license_url], "license_url": license_url.replace("http:", "https:"),
                     "key": key_name(key), "labels": len(labels)})
    return rows


def _annotations(root: Path) -> Path:
    target = root / "fsld-annotations.zip"
    if target.exists():
        content = target.read_bytes()
    else:
        with requests.get(ANNOTATIONS, stream=True, timeout=(10, 20)) as response:
            response.raise_for_status()
            content = response.raw.read(2_000_001)
    if len(content) > 2_000_000 or hashlib.md5(content).hexdigest() != ANNOTATIONS_MD5:
        raise ValueError("Freesound annotation archive failed its published checksum")
    if not target.exists():
        atomic_bytes(target, content)
    return target


def _inventory(root: Path, annotation_path: Path) -> list[dict]:
    with requests.Session() as session:
        stream, archive = open_zip(ARCHIVE, session, SIZE)
        with stream, archive, zipfile.ZipFile(annotation_path) as annotations:
            metadata_bytes = read_member(stream, archive, archive.getinfo("metadata.json"))
            metadata = json.loads(metadata_bytes)
            members = defaultdict(list)
            for name in archive.namelist():
                match = re.fullmatch(r"audio/[^/]+/(\d+)_\d+\.[^/]+\.wav", name)
                if match:
                    members[int(match[1])].append(name)
            rows = []
            for row in eligible(metadata, annotations):
                paths = members[row["id"]]
                if len(paths) == 1:
                    info = archive.getinfo(paths[0])
                    if info.file_size <= 12 * 1024**2:
                        rows.append(dict(row, member=info.filename, crc=info.CRC))
            atomic_json(root / "fsld-inventory.json", {
                "annotation_md5": ANNOTATIONS_MD5, "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
                "archive_size": SIZE, "rows": rows,
            })
    return rows


def extract(root: Path, workers: int, *, extended: bool = False) -> None:
    """Resume validated per-loop features without storing a full archive or audio."""
    root.mkdir(parents=True, exist_ok=True)
    annotation_path = _annotations(root)
    inventory = root / "fsld-inventory.json"
    rows = json.loads(inventory.read_text())["rows"] if inventory.exists() else _inventory(root, annotation_path)
    if extended:
        original = pd.read_parquet(root / "freesound.parquet")
        allowed_ids = set(original.id)
        rows = [row for row in rows if row["id"] in allowed_ids]
    print(f"{len(rows)} eligible licensed archive members", flush=True)
    state = local()
    signature = config_signature() + ("-harmonic-bass-v1" if extended else "")
    columns = HISTOGRAM
    if extended:
        from .research_features import release_histograms
        columns = HISTOGRAM + [f"{group}_{i}" for group in ("harmonic", "bass") for i in range(12)]

    def one(row):
        checkpoint = root / ("freesound-extended" if extended else "freesound") / f"{row['id']}.json"
        if checkpoint.exists():
            prior = json.loads(checkpoint.read_text())
            if prior["source"] == row and prior["config"] == signature:
                return prior
        if not hasattr(state, "archive"):
            state.session = requests.Session()
            state.stream, state.archive = open_zip(ARCHIVE, state.session, SIZE)
        info = state.archive.getinfo(row["member"])
        if info.CRC != row["crc"]:
            raise ValueError("Freesound archive member changed")
        content = read_member(state.stream, state.archive, info)
        with sf.SoundFile(io.BytesIO(content)) as audio:
            sr = audio.samplerate
            samples = audio.read(min(audio.frames, sr * 30), dtype="float32", always_2d=True).mean(axis=1)
        if samples.size < sr // 2:
            return None
        if sr != DEFAULT_CONFIG.sample_rate:
            import librosa

            samples = librosa.resample(samples, orig_sr=sr, target_sr=DEFAULT_CONFIG.sample_rate)
        if extended:
            histogram = release_histograms(samples, DEFAULT_CONFIG.sample_rate).ravel()
        else:
            histogram = histogram_from_audio(samples, DEFAULT_CONFIG.sample_rate)
        result = {"source": row, "config": signature, "hash": hashlib.sha256(content).hexdigest(),
                  **dict(zip(columns, histogram.tolist(), strict=True))}
        atomic_json(checkpoint, result)
        return result

    results, failed = [], Counter()

    def process(row):
        try:
            return one(row), None
        except (OSError, ValueError, RuntimeError, requests.RequestException, zipfile.BadZipFile) as exc:
            return None, f"{row['id']}: {exc}"

    with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (result, error) in enumerate(pool.map(process, rows), start=1):
            if result:
                results.append(result)
            if error:
                failed[error] += 1
            if i % 25 == 0 or i == len(rows):
                print(f"{i}/{len(rows)} loops checked, {len(results)} features, {sum(failed.values())} errors", flush=True)
    if not results:
        raise RuntimeError(f"no loop features could be extracted: {dict(failed)}")
    table = pd.DataFrame([{**row["source"], "sha256": row["hash"], **{k: row[k] for k in columns}}
                          for row in results]).drop_duplicates("sha256")
    features = table[columns].to_numpy().reshape(len(table), -1, 12)
    if not np.isfinite(features).all() or (features < 0).any() or not np.allclose(features.sum(axis=-1), 1):
        raise ValueError("invalid chroma values in the Freesound checkpoints")
    table_path = root / ("freesound-extended.parquet" if extended else "freesound.parquet")
    atomic_bytes(table_path, table.to_parquet(index=False))
    report = root / ("fsld-extended-extraction.json" if extended else "fsld-extraction.json")
    atomic_json(report, {"eligible": len(rows), "usable": len(table), "errors": dict(failed)})
    print(f"{len(table)} distinct loops; features saved to {table_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=ROOT)
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 9))
    parser.add_argument("--extended", action="store_true", help="Also extract harmonic and bass profiles.")
    args = parser.parse_args()
    extract(args.work, args.workers, extended=args.extended)


if __name__ == "__main__":
    main()
