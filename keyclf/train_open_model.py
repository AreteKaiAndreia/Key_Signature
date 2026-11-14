"""Train on original synthetic audio and licensed loops, and freeze before testing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from .experiments import rotate_training
from .features import DEFAULT_CONFIG, config_signature, histogram_from_audio
from .fmak import atomic_json, digest
from .fmak_experiment import file_hash, split_ids, xy
from .keys import KEY_NAMES, parse_key
from .open_loops import HISTOGRAM, ROOT
from .open_model import FORMAT, POWERS, OpenKeyModel, transform

VERSION = "licensed-loop-and-original-synth-v1"
OUTPUT = Path("reports/open-key")
GS = Path("data/features/giantsteps_features.parquet")
FMAK = Path("data/work/fmak-pilot/features.parquet")
PINNED_TESTS = Path("reports/hooktheory/selection.json")
SYNTH_COUNT = 480
SEED = 1242
SCALES = ((0, 2, 4, 5, 7, 9, 11), (0, 2, 3, 5, 7, 8, 10))
PROGRESSIONS = (
    ((0, 4, 5, 3), (0, 5, 3, 4), (0, 3, 4, 0), (5, 3, 0, 4), (0, 1, 3, 4)),
    ((0, 5, 2, 6), (0, 6, 5, 6), (0, 3, 4, 0), (5, 6, 0, 0), (0, 2, 3, 4)),
)


def synthesize(index: int) -> tuple[np.ndarray, int]:
    """Create a seeded original phrase with bass, harmony, and optional melody."""
    rng = np.random.default_rng(SEED + index)
    sr = DEFAULT_CONFIG.sample_rate
    mode = index % 2
    tonic = int(rng.integers(0, 12))
    scale = SCALES[mode]
    chords = PROGRESSIONS[mode][int(rng.integers(len(PROGRESSIONS[mode])))]
    beat = 60 / rng.uniform(78, 155)
    bar = round(4 * beat * sr)
    audio = np.zeros(bar * len(chords), dtype=np.float32)
    harmonic_gain = rng.uniform(0.15, 1.0, 4) * (1, .5, .32, .18)
    mix = rng.dirichlet([2, 2, 1.7])
    if rng.random() < .22:
        mix[2] = 0
    tuned = rng.uniform(-.2, .2)

    def add(note, begin, length, gain):
        start = round(begin * bar)
        n = min(round(length * bar), len(audio) - start)
        t = np.arange(n) / sr
        phase = t * (2 * np.pi * 440 * 2 ** ((note - 69 + tuned) / 12))
        envelope = np.minimum(1, np.minimum(t / .012, (t[-1] - t) / .035)) if n > 1 else np.zeros(n)
        signal = sum(amplitude * np.sin(overtone * phase)
                     for overtone, amplitude in enumerate(harmonic_gain, 1))
        audio[start:start + n] += gain * envelope * signal

    for bar_index, degree in enumerate(chords):
        root = scale[degree]
        for pulse in range(4):
            bass_note = root if pulse < 3 or rng.random() < .55 else (root + 7) % 12
            add(36 + tonic + bass_note, bar_index + pulse / 4, rng.uniform(.08, .24), mix[0])
        for third in (0, 2, 4):
            step = degree + third
            note = 48 + tonic + scale[step % 7] + 12 * (step // 7)
            if mode and degree == 4 and third == 2:
                note += int(rng.random() < .75)
            add(note, bar_index, rng.uniform(.65, .99), mix[1] / 2)
        for pulse in range(4):
            if rng.random() < .18:
                continue
            step = int(rng.choice([degree % 7, (degree + 2) % 7, (degree + 4) % 7, rng.integers(7)]))
            add(60 + tonic + scale[step], bar_index + pulse / 4, rng.uniform(.08, .24), mix[2])
    audio += rng.normal(0, .006, len(audio)).astype(np.float32)
    return audio, tonic + mode * 12


def synthetic_features(work: Path) -> tuple[np.ndarray, np.ndarray]:
    """Resume the original synthetic corpus from per-phrase chroma files."""
    rows, keys = [], []
    for index in range(SYNTH_COUNT):
        path = work / "synthetic" / f"{index}.json"
        if path.exists():
            item = json.loads(path.read_text())
        else:
            item = {}
        expected = [VERSION, config_signature(), SEED + index]
        if item.get("signature") != expected:
            audio, key = synthesize(index)
            histogram = histogram_from_audio(audio, DEFAULT_CONFIG.sample_rate)
            item = {"signature": expected, "key": key, "histogram": histogram.tolist()}
            atomic_json(path, item)
        rows.append(item["histogram"])
        keys.append(item["key"])
        if (index + 1) % 120 == 0:
            print(f"{index + 1}/{SYNTH_COUNT} synthetic phrases ready", flush=True)
    return np.asarray(rows), np.asarray(keys)


def measured(model: OpenKeyModel, frame: pd.DataFrame) -> dict:
    """Apply production ranking and report counts per immutable partition."""
    x, keys = xy(frame)
    probabilities = model.predict_proba(x)
    orders = np.argsort(-probabilities, axis=1, kind="stable")
    first = int(np.sum(orders[:, 0] == keys))
    three = int(np.sum(np.any(orders[:, :3] == keys[:, None], axis=1)))
    true_scores = np.maximum(probabilities[np.arange(len(keys)), keys], np.finfo(float).tiny)
    return {"n": len(keys), "first": first, "top3": three, "first_rate": first / len(keys),
            "top3_rate": three / len(keys), "log_loss": float(-np.log(true_scores).mean())}


def score(results: dict) -> tuple:
    """Favor candidates closest to passing both domains, then mean accuracy."""
    values = list(results.values())
    margin = min(min(r["first_rate"] - .5, r["top3_rate"] - .7) for r in values)
    return (margin, sum(r["first_rate"] for r in values), sum(r["top3_rate"] for r in values),
            -sum(r["log_loss"] for r in values))


def train(work: Path, out: Path, selection_only: bool) -> None:
    """Selection reads validation results. Final evaluation runs only on demand."""
    benches = {"giantsteps": pd.read_parquet(GS), "fmak": pd.read_parquet(FMAK)}
    hashes = {name: digest({"ids": split_ids(frame)["test"]}) for name, frame in benches.items()}
    expected = json.loads(PINNED_TESTS.read_text())["test_ids_sha256"]
    if hashes != expected:
        raise ValueError("test partition IDs differ from the earlier frozen manifest")
    validation = {name: data.loc[data.split == "val"] for name, data in benches.items()}
    loops_path = work / "freesound.parquet"
    loops = pd.read_parquet(loops_path)
    source = {name: file_hash(path) for name, path in (("loops", loops_path), ("giantsteps", GS), ("fmak", FMAK))}
    artifact = out / "open-key-model.json"
    receipt = out / "selection.json"
    if not receipt.exists():
        sx, sy = synthetic_features(work)
        lx, ly = xy(loops)
        candidates = {}
        selected = None
        for corpus, x, y in (("synthetic", sx, sy), ("loops", lx, ly),
                             ("combined", np.vstack((lx, sx)), np.r_[ly, sy])):
            x, y = rotate_training(x, y)
            for power in POWERS:
                features = transform(x, power)
                for c in (1., 10., 100.):
                    estimator = LogisticRegression(C=c, class_weight="balanced", solver="lbfgs", max_iter=500)
                    estimator.fit(features, y)
                    profiles = np.array([np.mean([np.roll(estimator.coef_[mode * 12 + tonic], -tonic)
                                                  for tonic in range(12)], axis=0) for mode in range(2)])
                    offsets = estimator.intercept_.reshape(2, 12).mean(axis=1)
                    for weight in (0., 2., 5.):
                        label = f"{corpus}, power={power:g}, C={c:g}, KS={weight:g}"
                        model = OpenKeyModel(profiles, offsets, power, weight)
                        metrics = {name: measured(model, data) for name, data in validation.items()}
                        candidates[label] = metrics
                        if selected is None or score(metrics) > score(candidates[selected]):
                            selected = label
                            best = model
        model_data = {"format": FORMAT, "version": VERSION, "classes": list(KEY_NAMES),
                      "profiles": best.profiles.tolist(), "mode_offsets": best.mode_offsets.tolist(),
                      "power": best.power, "ks_weight": best.ks_weight, "variant": selected}
        atomic_json(artifact, model_data)
        atomic_json(receipt, {"version": VERSION, "selected": selected, "validation": candidates,
                              "inputs": source, "test_ids_sha256": hashes, "model_sha256": file_hash(artifact)})
    frozen = json.loads(receipt.read_text())
    if frozen["inputs"] != source or frozen["model_sha256"] != file_hash(artifact):
        raise ValueError("frozen experiment inputs or final model changed")
    print(json.dumps({"selected": frozen["selected"], "validation": frozen["validation"][frozen["selected"]]}, indent=2))
    if not selection_only:
        model = OpenKeyModel.load(artifact)
        metrics = {name: measured(model, data.loc[data.split == "test"]) for name, data in benches.items()}
        passed = all(v["first_rate"] >= .5 and v["top3_rate"] >= .7 for v in metrics.values())
        outcome = {"test": metrics, "passed": passed, "model_sha256": frozen["model_sha256"]}
        atomic_json(out / "test.json", outcome)
        print(json.dumps(outcome, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--final", action="store_true", help="Score only the frozen candidate on held-out rows.")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        train(args.work, args.out, selection_only=not args.final)


if __name__ == "__main__":
    main()
