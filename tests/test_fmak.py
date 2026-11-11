"""Offline checks for bounded FMAK extraction and leakage-safe comparison."""

import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from keyclf import fmak, fmak_experiment, rangezip
from keyclf.keys import KEY_NAMES


class MemoryResponse:
    def __init__(self, status, content, headers):
        self.status_code, self.headers = status, headers
        self.raw = io.BytesIO(content)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None


class MemorySession:
    def __init__(self, data):
        self.data = data
        self.ranges = []

    def get(self, url, *, headers, **kwargs):
        start, stop = map(int, headers["Range"].removeprefix("bytes=").split("-"))
        self.ranges.append((start, stop))
        return MemoryResponse(206, self.data[start:stop + 1], {
            "Content-Range": f"bytes {start}-{stop}/{len(self.data)}",
        })


def test_range_reads_refuse_full_download_or_mismatched_bytes():
    for response in (MemoryResponse(200, b"whole archive", {}),
                     MemoryResponse(206, b"wrong", {"Content-Range": "bytes 4-8/10"})):
        class Session:
            def get(self, *args, **kwargs):
                return response
        with pytest.raises(ValueError):
            rangezip.fetch_range(Session(), "https://test.invalid", 0, 5)
        assert response.raw.tell() == 0


def test_zip_reads_selected_member_and_checks_crc():
    buffer = io.BytesIO()
    audio = b"unique audio marker"
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("000/000001.mp3", audio)
        z.writestr("000/000002.mp3", b"x" * (1024**2))
    data = buffer.getvalue()
    session = MemorySession(data)
    reader, z = rangezip.open_zip("https://test.invalid", session, len(data))
    assert rangezip.read_member(reader, z, z.getinfo("000/000001.mp3")) == audio
    assert sum(stop - start + 1 for start, stop in session.ranges) < len(data)
    corrupt = data.replace(audio, b"y" * len(audio), 1)
    reader, z = rangezip.open_zip("https://test.invalid", MemorySession(corrupt), len(corrupt))
    with pytest.raises(zipfile.BadZipFile, match="CRC"):
        rangezip.read_member(reader, z, z.getinfo("000/000001.mp3"))


def test_pilot_fixed_count_and_artist_isolation():
    pool = [{"track_id": f"fmak-{i:06d}", "key_id": i % 24, "artist_id": i // 12, "archive": "000-019.zip"}
            for i in range(1440)]
    plan = fmak.select_pilot(pool, 500)
    assert plan == fmak.select_pilot(pool[::-1], 500)
    assert Counter(row["split"] for row in plan) == {"train": 350, "val": 75, "test": 75}
    assert sum(row["key_id"] < 12 for row in plan) == 300
    for artist in {row["artist_id"] for row in plan}:
        assert len({row["split"] for row in plan if row["artist_id"] == artist}) == 1


def test_label_and_license_validation():
    csv = b"track_id,key_and_mode\n1,Db Major\n2,A Minor\n3,unknown\n"
    assert fmak.read_labels(csv) == {1: 1, 2: 21, 3: None}
    with pytest.raises(ValueError, match="duplicate"):
        fmak.read_labels(csv + b"1,C Major\n")
    assert fmak.allowed_license("Attribution-NonCommercial-ShareAlike 3.0 Unported")
    assert not fmak.allowed_license("Attribution-NoDerivs 3.0 Unported")
    assert not fmak.allowed_license("All Rights Reserved")


def test_extract_resumes_and_removes_temporary_audio(tmp_path, monkeypatch):
    buffer = io.BytesIO()
    shard = "000-019.zip"
    tracks = []
    with zipfile.ZipFile(buffer, "w") as z:
        for i, split in enumerate(("train", "val", "test")):
            path, audio = f"000/{i:06d}.mp3", f"synthetic clip {i}".encode()
            z.writestr(path, audio)
            info = z.getinfo(path)
            tracks.append({"track_id": f"fmak-{i:06d}", "key": KEY_NAMES[i], "split": split,
                           "artist_id": i, "archive": shard, "member": path,
                           "crc32": f"{info.CRC:08x}", "audio_bytes": info.file_size})
    data = buffer.getvalue()
    manifest = {"protocol": fmak.PROTOCOL, "tracks": tracks, "features": fmak.config_signature(),
                "archives": {shard: {"size": len(data)}}}
    fmak.atomic_json(tmp_path / "manifest.json", dict(manifest, signature=fmak.digest(manifest)))
    monkeypatch.setattr(fmak, "original_checksums", lambda _: set())
    monkeypatch.setattr(fmak.requests, "Session", lambda: MemorySession(data))
    paths = []
    def features(path):
        paths.append(Path(path))
        assert Path(path).read_bytes().startswith(b"synthetic clip ")
        return np.r_[np.zeros(24), np.ones(12) / 12]
    monkeypatch.setattr(fmak, "extract_features", features)
    assert fmak.extract(tmp_path, tmp_path, 1, limit=1) == 0
    assert not (tmp_path / "features.parquet").exists()
    assert fmak.extract(tmp_path, tmp_path, 1) == 0
    assert len(paths) == 3 and not any(path.exists() for path in paths)
    summary = json.loads((tmp_path / "extraction.json").read_text())
    assert summary["counts"] == {"resumed": 1, "extracted": 2}
    cache = tmp_path / "features.parquet"
    assert len(pd.read_parquet(cache)) == 3
    before = cache.read_bytes()
    assert fmak.extract(tmp_path, tmp_path, 2) == 0
    assert cache.read_bytes() == before and len(paths) == 3


def test_comparison_freezes_validation_before_test(tmp_path, monkeypatch):
    config = {"name": "histogram_rf_transpose", "group": "histogram", "blocks": ["hist"],
              "algorithm": "forest", "augment": True, "leaf": 2, "max_features": "sqrt"}
    parts = ("train", "train", "val", "test")
    original = []
    tracks, new = [], []
    for i, split in enumerate(parts):
        x = np.eye(12)[i]
        original.append({"track_id": str(i), "key": KEY_NAMES[i], "split": split,
                         **dict(zip(fmak.HISTOGRAM, x, strict=True))})
        tracks.append({"track_id": f"fmak-{i:06d}", "key": KEY_NAMES[i + 12], "split": split,
                       "artist_id": i, "crc32": f"{i:08x}"})
    manifest = {"protocol": fmak.PROTOCOL, "tracks": tracks}
    signature = fmak.digest(manifest)
    for i, track in enumerate(tracks):
        new.append(dict(track, manifest=signature, md5=f"{i:032x}",
                        **dict(zip(fmak.HISTOGRAM, np.eye(12)[i], strict=True))))
    fmak.atomic_json(tmp_path / "manifest.json", dict(manifest, signature=signature))
    base, additional = pd.DataFrame(original), pd.DataFrame(new)
    base.to_parquet(tmp_path / "base.parquet")
    additional.to_parquet(tmp_path / "features.parquet")
    monkeypatch.setattr(fmak_experiment, "COUNTS", {"train": 2, "val": 1, "test": 1})
    monkeypatch.setattr(fmak_experiment, "estimator_for", lambda _: RandomForestClassifier(n_estimators=4, random_state=42))
    base_fit = fmak_experiment.fit(base[base.split != "test"], additional.iloc[:0], config, 1)
    reference = tmp_path / "reference.joblib"
    joblib.dump({"model": base_fit, "feature_names": fmak.HISTOGRAM, "classes": KEY_NAMES, "config": config,
                 "splits": fmak_experiment.split_ids(base)}, reference)
    output = tmp_path / "report"
    real_scores = fmak_experiment.scores
    def checked_scores(model, frame):
        if "test" in set(frame.split):
            assert (output / "selection.json").exists()
        return real_scores(model, frame)
    monkeypatch.setattr(fmak_experiment, "scores", checked_scores)
    args = (tmp_path, tmp_path / "base.parquet", reference, output, tmp_path / "expanded.joblib")
    result = fmak_experiment.run(*args)
    assert result["training_recordings"] == {"giantsteps": 3, "fmak": 3}
    assert result["test_used_for_selection"] is False
    assert fmak_experiment.run(*args) == result
