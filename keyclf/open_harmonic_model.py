"""Fit and verify a rotation-equivariant key model with bass and harmony."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from .baseline import ks_scores
from .cloud_audio import SAMPLE_RATE
from .experiments import rotate_training
from .fmak import atomic_json, digest
from .fmak_experiment import file_hash, split_ids
from .keys import KEY_NAMES, parse_key
from .open_benchmark import COLUMNS
from .open_loops import ROOT
from .research_features import release_histograms
from .train_open_model import (
    FMAK, GS, PINNED_TESTS, SYNTH_COUNT, score, synthesize,
)

FORMAT = "open-harmonic-key-v1"
OUT = Path("reports/open-key-harmonic")
GS_RESEARCH = Path("data/work/research_features.parquet")


@dataclass(frozen=True)
class HarmonicKeyModel:
    """Learn two sets of pitch weights and rotate them across tonics."""

    weights: np.ndarray
    bias: np.ndarray
    power: float
    ks_weight: float

    def learned_logits(self, features: np.ndarray) -> np.ndarray:
        blocks = np.asarray(features, dtype=float).reshape(-1, 3, 12)
        transformed = blocks ** self.power
        transformed /= transformed.sum(axis=-1, keepdims=True)
        rotated = np.stack([np.roll(w, tonic, axis=1).ravel() for w in self.weights for tonic in range(12)])
        return transformed.reshape(-1, 36) @ rotated.T + np.repeat(self.bias, 12)

    def scores(self, features: np.ndarray) -> np.ndarray:
        logits = self.learned_logits(features)
        if self.ks_weight:
            profiles = np.asarray(features, dtype=float).reshape(-1, 3, 12)[:, 0]
            logits += self.ks_weight * ks_scores(profiles)
        return softmax(logits, axis=1)

    @classmethod
    def load(cls, path: Path) -> HarmonicKeyModel:
        value = json.loads(path.read_text())
        weights = np.asarray(value["weights"], dtype=float)
        bias = np.asarray(value["bias"], dtype=float)
        power, ks_weight = value["power"], value["ks_weight"]
        if (value["format"] != FORMAT or tuple(value["classes"]) != KEY_NAMES
                or weights.shape != (2, 3, 12) or bias.shape != (2,)
                or not np.isfinite(weights).all() or not np.isfinite(bias).all()
                or power not in (.5, 1., 2.) or not 0 <= ks_weight <= 10):
            raise ValueError("Invalid harmonic key coefficients.")
        return cls(weights, bias, power, ks_weight)


def synthetic_blocks(work: Path) -> tuple[np.ndarray, np.ndarray]:
    """Render original phrases and save only the three pitch summaries."""
    vectors, keys = [], []
    for index in range(SYNTH_COUNT):
        path = work / "synthetic-extended" / f"{index}.json"
        if path.exists():
            item = json.loads(path.read_text())
        else:
            samples, key = synthesize(index)
            item = {"key": key, "features": release_histograms(samples, SAMPLE_RATE).ravel().tolist()}
            atomic_json(path, item)
        vectors.append(item["features"])
        keys.append(item["key"])
        if (index + 1) % 120 == 0:
            print(f"{index + 1}/{SYNTH_COUNT} synthetic harmonic/bass features ready", flush=True)
    return np.asarray(vectors), np.asarray(keys)


def metrics(model: HarmonicKeyModel, x: np.ndarray, targets: np.ndarray) -> dict:
    probabilities = model.scores(x)
    ranked = np.argsort(-probabilities, axis=1, kind="stable")
    first = int((ranked[:, 0] == targets).sum())
    top3 = int(np.any(ranked[:, :3] == targets[:, None], axis=1).sum())
    n = len(targets)
    correct_scores = probabilities[np.arange(n), targets]
    return {"n": n, "first": first, "top3": top3, "first_rate": first / n, "top3_rate": top3 / n,
            "log_loss": float(-np.log(np.maximum(correct_scores, np.finfo(float).tiny)).mean())}


def key_ids(series: pd.Series) -> np.ndarray:
    return np.asarray([parse_key(label) for label in series])


def xy(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return frame[COLUMNS].to_numpy(dtype=float), key_ids(frame.key)


def benchmark(work: Path) -> dict[str, pd.DataFrame]:
    """Reject any change to labels, recording IDs, or benchmark group assignment."""
    sources = {name: pd.read_parquet(path) for name, path in (("giantsteps", GS), ("fmak", FMAK))}
    hashes = {name: digest({"ids": split_ids(frame)["test"]}) for name, frame in sources.items()}
    if hashes != json.loads(PINNED_TESTS.read_text())["test_ids_sha256"]:
        raise ValueError("Frozen test IDs changed.")
    benches = {"giantsteps": pd.read_parquet(GS_RESEARCH), "fmak": pd.read_parquet(work / "fmak-bench.parquet")}
    for name, frame in benches.items():
        reference = sources[name]
        columns = ["track_id", "key", "split"]
        actual = frame.loc[frame.split.isin(["val", "test"]), columns].sort_values("track_id").reset_index(drop=True)
        expected = reference.loc[reference.split.isin(["val", "test"]), columns].sort_values("track_id").reset_index(drop=True)
        if not actual.equals(expected):
            raise ValueError(f"Benchmark rows, labels, or splits changed: {name}")
    return benches


def run(work: Path, out: Path, final: bool) -> None:
    benches = benchmark(work)
    loops_path = work / "freesound-extended.parquet"
    source_files = [loops_path, GS, FMAK, GS_RESEARCH, work / "fmak-bench.parquet", PINNED_TESTS]
    source_hashes = {str(path): file_hash(path) for path in source_files}
    receipt, artifact = out / "selection.json", out / "open-key-model.json"
    if not receipt.exists():
        loops = pd.read_parquet(loops_path)
        lx, ly = xy(loops)
        sx, sy = synthetic_blocks(work)
        # Use the larger available GiantSteps development partition for
        # comparison. Neither benchmark contributes any fitted examples.
        validation = {"giantsteps": xy(benches["giantsteps"].loc[benches["giantsteps"].split != "test"]),
                      "fmak": xy(benches["fmak"].loc[benches["fmak"].split == "val"])}
        candidates = {}
        best_name = None
        for corpus, x, y in (("loops", lx, ly), ("synthetic", sx, sy),
                             ("combined", np.vstack((lx, sx)), np.r_[ly, sy])):
            augmented, labels = rotate_training(x, y)
            print(f"Fitting {corpus} on {len(x)} examples", flush=True)
            for power in (.5, 1., 2.):
                blocks = augmented.reshape(-1, 3, 12) ** power
                blocks /= blocks.sum(axis=-1, keepdims=True)
                for c in (1., 10., 100.):
                    estimator = LogisticRegression(C=c, class_weight="balanced", max_iter=600)
                    estimator.fit(blocks.reshape(-1, 36), labels)
                    aligned = np.stack([np.roll(coef.reshape(3, 12), -(key % 12), axis=1)
                                        for key, coef in enumerate(estimator.coef_)])
                    weights = aligned.reshape(2, 12, 3, 12).mean(axis=1)
                    bias = estimator.intercept_.reshape(2, 12).mean(axis=1)
                    for bass_weight in (0., .5, 1., 1.5):
                        adjusted = weights.copy()
                        adjusted[:, 2] *= bass_weight
                        for ks_weight in (0., 2., 5., 9.):
                            for minor_offset in (0., .5, 1.):
                                name = (f"{corpus}, power={power:g}, C={c:g}, bass={bass_weight:g}, "
                                        f"KS={ks_weight:g}, minor={minor_offset:g}")
                                model = HarmonicKeyModel(adjusted, bias + [0, minor_offset], power, ks_weight)
                                measured = {key: metrics(model, *rows) for key, rows in validation.items()}
                                candidates[name] = measured
                                if best_name is None or score(measured) > score(candidates[best_name]):
                                    best_name, best = name, model
        value = {"format": FORMAT, "classes": list(KEY_NAMES), "weights": best.weights.tolist(),
                 "bias": best.bias.tolist(), "power": best.power, "ks_weight": best.ks_weight,
                 "variant": best_name}
        atomic_json(artifact, value)
        atomic_json(receipt, {"inputs": source_hashes, "validation": candidates, "selected": best_name,
                              "model_sha256": file_hash(artifact)})
    selection = json.loads(receipt.read_text())
    if selection["inputs"] != source_hashes or selection["model_sha256"] != file_hash(artifact):
        raise ValueError("Frozen selection sources or model changed.")
    print(json.dumps({"selected": selection["selected"],
                      "validation": selection["validation"][selection["selected"]]}, indent=2))
    if final:
        model = HarmonicKeyModel.load(artifact)
        test = {name: metrics(model, *xy(frame.loc[frame.split == "test"])) for name, frame in benches.items()}
        outcome = {"test": test, "passed": all(value["first_rate"] >= .5 and value["top3_rate"] >= .7
                                               for value in test.values()),
                   "model_sha256": selection["model_sha256"]}
        atomic_json(out / "test.json", outcome)
        print(json.dumps(outcome, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.work, args.out, args.final)


if __name__ == "__main__":
    main()
