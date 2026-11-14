"""Add harmonic and bass summaries to the existing evaluation-only partitions."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from threading import local

import numpy as np
import pandas as pd
import requests
from threadpoolctl import threadpool_limits

from .cloud_audio import decode_clip
from .fmak import archive_url, atomic_bytes, atomic_json, load_manifest
from .open_loops import ROOT
from .rangezip import open_zip, read_member
from .research_features import release_histograms

FMA_ROOT = Path("data/work/fmak-pilot")
COLUMNS = [f"{block}_{i}" for block in ("hist", "harmonic", "bass") for i in range(12)]
FEATURE_VERSION = "harmonic-bass-120s-v1"


def extract_fmak(work: Path = ROOT, workers: int = 4) -> None:
    """Stream the frozen FMAK validation and test tracks without retaining audio."""
    manifest = load_manifest(FMA_ROOT)
    reference = pd.read_parquet(FMA_ROOT / "features.parquet").set_index("track_id")
    rows = [row for row in manifest["tracks"] if row["split"] in {"val", "test"}]
    rows.sort(key=lambda row: (row["archive"], row["track_id"]))
    state = local()

    def extract(track: dict) -> dict:
        track_id = track["track_id"]
        expected_md5 = reference.loc[track_id, "md5"]
        signature = [FEATURE_VERSION, manifest["signature"], expected_md5]
        path = work / "fmak-bench" / f"{track_id}.json"
        if path.exists():
            result = json.loads(path.read_text())
            if result["signature"] == signature:
                return result["row"]
        shard = track["archive"]
        if getattr(state, "shard", None) != shard:
            if hasattr(state, "session"):
                state.archive.close()
                state.stream.close()
            else:
                state.session = requests.Session()
            state.stream, state.archive = open_zip(archive_url(shard), state.session, manifest["archives"][shard]["size"])
            state.shard = shard
        info = state.archive.getinfo(track["member"])
        if f"{info.CRC:08x}" != track["crc32"] or info.file_size != track["audio_bytes"]:
            raise ValueError(f"FMAK member changed for {track_id}")
        content = read_member(state.stream, state.archive, info)
        if hashlib.md5(content).hexdigest() != expected_md5:
            raise ValueError(f"FMAK track identity changed for {track_id}")
        samples, _ = decode_clip(content, "benchmark.mp3")
        features = release_histograms(samples, 22050).ravel()
        if not np.isfinite(features).all():
            raise ValueError(f"Invalid benchmark features for {track_id}")
        row = {"track_id": track_id, "split": track["split"], "key": track["key"],
               **dict(zip(COLUMNS, features.tolist(), strict=True))}
        atomic_json(path, {"signature": signature, "row": row})
        return row

    outcomes = []
    with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=workers) as pool:
        for index, row in enumerate(pool.map(extract, rows), start=1):
            outcomes.append(row)
            if index % 10 == 0:
                print(f"{index}/{len(rows)} FMAK benchmark tracks ready", flush=True)
    table = pd.DataFrame(outcomes)
    atomic_bytes(work / "fmak-bench.parquet", table.to_parquet(index=False))


if __name__ == "__main__":
    extract_fmak()
