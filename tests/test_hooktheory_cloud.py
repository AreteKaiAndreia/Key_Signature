"""Regression tests for the notation pilot and release-day inference path."""

import copy
import io
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf
from sklearn.ensemble import RandomForestClassifier

from keyclf import cloud_audio, cloud_engines, features, hooktheory
from keyclf.cloud_engines import custom_scores, essentia_scores, native_key, rank_three
from keyclf.keys import KEY_NAMES


def chord(seconds=2):
    t = np.arange(round(seconds * cloud_audio.SAMPLE_RATE)) / cloud_audio.SAMPLE_RATE
    return .2 * sum(np.sin(2 * np.pi * hz * t) for hz in (261.626, 329.628, 391.995))


def wav(samples):
    buffer = io.BytesIO()
    sf.write(buffer, samples, cloud_audio.SAMPLE_RATE, format="WAV")
    return buffer.getvalue()


def test_direct_histogram_keeps_feature_values(tmp_path, monkeypatch):
    path = tmp_path / "chord.wav"
    path.write_bytes(wav(chord()))
    original = features.extract_features(path)[24:]
    monkeypatch.setattr(features, "first_last_bins", lambda *_: pytest.fail("computed edge features"))
    np.testing.assert_array_equal(features.extract_histogram(path), original)


def notation(tonic=0):
    return {
        "keys": [{"beat": 0, "tonic_pitch_class": tonic, "scale_degree_intervals": hooktheory.MAJOR}],
        "num_beats": 4,
        "melody": [{"onset": 0, "offset": 4, "pitch_class": 0, "octave": 0}],
        "harmony": [{"onset": 0, "offset": 4, "root_pitch_class": 0, "root_position_intervals": [4, 3]}],
    }


def row(artist, song, video, split="TRAIN"):
    return {"split": split, "hooktheory": {"artist": artist, "song": song},
            "youtube": {"id": video}, "annotations": notation()}


def test_render_ignores_key_labels_and_rejects_key_changes():
    first, second = notation(), notation(7)
    assert hooktheory.annotated_key(first) == 0
    assert hooktheory.annotated_key(second) == 7
    np.testing.assert_array_equal(hooktheory.render_audio(*hooktheory.note_events(first)),
                                  hooktheory.render_audio(*hooktheory.note_events(second)))
    second["keys"][0]["scale_degree_intervals"] = hooktheory.MINOR[0]
    assert hooktheory.annotated_key(second) == 19
    second["keys"].append(copy.deepcopy(first["keys"][0]))
    with pytest.raises(ValueError, match="not one global key"):
        hooktheory.annotated_key(second)


def test_pilot_excludes_duplicates_and_restarts_after_interruption(tmp_path, monkeypatch):
    data = {"a": row("Thé Band", "A Song", "1"), "b": row("Band", "A Song (live)", "2"),
            "c": row("Different Name", "Another Song", "1"),
            "d": row("Held Out", "Test Song", "3", "TEST"), "e": row("Artist", "Renamed", "3"),
            "f": row("Known", "Overlap", "4"), "g": row("New", "Song", "5")}
    excluded = {("known", "overlap")}
    candidates, audit = hooktheory.candidate_rows(data, excluded)
    assert [r["record_id"] for r in candidates] == ["a", "g"]
    assert audit == {"not_train": 1, "duplicate": 2, "excluded_identity": 2}
    monkeypatch.setattr(hooktheory, "acquire_release", lambda _: data)
    monkeypatch.setattr(hooktheory, "fma_identities", lambda *_: set(excluded))
    h = np.arange(1, 13, dtype=float)
    h /= h.sum()
    outcomes = iter([h, KeyboardInterrupt()])

    def extract(*_):
        outcome = next(outcomes)
        if isinstance(outcome, KeyboardInterrupt):
            raise outcome
        return outcome

    monkeypatch.setattr(hooktheory, "histogram_from_audio", extract)
    with pytest.raises(KeyboardInterrupt):
        hooktheory.create_pilot(tmp_path, tmp_path / "fma", 2)
    assert len(list(tmp_path.glob("checkpoints/*.json"))) == 1
    monkeypatch.setattr(hooktheory, "histogram_from_audio", lambda *_: h)
    resumed = hooktheory.create_pilot(tmp_path, tmp_path / "fma", 2)
    assert resumed["counts"] == {"resumed": 1, "rendered": 1}
    assert not [p for p in tmp_path.rglob("*") if p.suffix in {".wav", ".mp3", ".mp4"}]


@pytest.fixture(scope="module")
def forest():
    rng = np.random.default_rng(42)
    return RandomForestClassifier(n_estimators=4, random_state=42).fit(rng.random((48, 12)), np.tile(np.arange(24), 2))


@pytest.mark.parametrize("extension", ["wav", "mp3", "mp4"])
def test_both_rankings_on_recording_and_upload_formats(tmp_path, forest, extension):
    pytest.importorskip("essentia")
    assert shutil.which("ffmpeg"), "ffmpeg is required for the audio upload tests"
    source = tmp_path / "recording.wav"
    source.write_bytes(wav(chord()))
    target = tmp_path / f"clip.{extension}"
    if extension == "wav":
        target = source
    else:
        codec = "aac" if extension == "mp4" else "libmp3lame"
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(source), "-c:a", codec, str(target)], check=True)
    samples, clipped = cloud_audio.decode_clip(target.read_bytes(), target.name)
    assert not clipped
    profile, scores = essentia_scores(samples)
    native, strength = native_key(profile)
    assert KEY_NAMES[native] == rank_three(scores)[0][0] == "C major"
    assert strength == pytest.approx(rank_three(scores)[0][1], abs=1e-5)
    histogram, probabilities = custom_scores(forest, samples)
    assert histogram.sum() == pytest.approx(1.0)
    for vector in (scores, probabilities):
        ranked = rank_three(vector)
        assert len({k for k, _ in ranked}) == 3
        assert [s for _, s in ranked] == sorted((s for _, s in ranked), reverse=True)


def test_decode_rejects_bad_audio_and_limits_duration(monkeypatch):
    decode = cloud_audio.decode_clip
    with pytest.raises(ValueError, match="readable audio"):
        decode(b"bad audio", "broken.mp3")
    with pytest.raises(ValueError, match="silent"):
        decode(wav(np.zeros(cloud_audio.SAMPLE_RATE)), "silence.wav")
    # Narrow the limits for a quick deterministic test of the same FFmpeg path.
    monkeypatch.setattr(cloud_audio, "MAX_SECONDS", 1)
    samples, clipped = decode(wav(chord()), "long.wav")
    assert clipped and len(samples) == cloud_audio.SAMPLE_RATE
    monkeypatch.setattr(cloud_audio, "MAX_BYTES", 12)
    with pytest.raises(ValueError, match="50 MB"):
        decode(b"x" * 13, "large.mp4")


def test_release_checks_checksum_before_loading_and_rejects_missing_weights(tmp_path, monkeypatch):
    class Response:
        url = "https://release-assets.githubusercontent.com/local-test-joblib"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def raise_for_status(self):
            pass

        def iter_content(self, _):
            return [b"this is not the pinned release"]

    monkeypatch.setattr(cloud_engines.requests, "get", lambda *_a, **_kw: Response())
    monkeypatch.setattr(cloud_engines.joblib, "load", lambda _: pytest.fail("loaded unverified weights"))
    with pytest.raises(ValueError, match="checksum"):
        cloud_engines.released_model(f"{cloud_engines.REPO}/releases/download/v1/model.joblib", "0" * 64)
    with pytest.raises(FileNotFoundError, match="Train the local"):
        cloud_engines.local_model(tmp_path / "missing.joblib")
