"""Export aggregate first and top-three metrics from the pinned benchmark clips."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from keyclf.fmak import atomic_json
from keyclf.keys import KEY_NAMES, relative_key
from keyclf.open_benchmark import FMA_ROOT
from keyclf.open_ensemble import ENGINES, SoftVotingKeyModel, open_feature_scorers
from keyclf.open_harmonic_model import benchmark, xy
from keyclf.open_loops import ROOT
from keyclf.relative_pairs import rank_relative_pairs

CACHE = Path("data/work/model-diagnostics/essentia")
PUBLIC = Path("keyclf")
ESSENTIA_VERSION = "cloud-hpcp-test-v1-essentia-2.1-beta6-dev"
METHODS = {
    "current": ("neural_network", "Neural Network"),
    "linear": ("linear_svm", "Linear SVM"),
    "forest": ("random_forest", "Random Forest"),
    "boosting": ("gradient_boosting", "Gradient Boosting"),
    "essentia": ("essentia", "Essentia"),
    "ensemble": ("soft_voting", "Soft voting (five engines)"),
}
PAIR_BY_KEY = np.array([min(key, relative_key(key)) for key in range(len(KEY_NAMES))])


def clip_metrics(truth: np.ndarray, predictions: np.ndarray, classes: int) -> dict:
    """Measure list hits and macro recall and precision for distinct predictions."""
    matches = np.any(predictions == truth[:, None], axis=1)
    true_positives = np.bincount(truth[matches], minlength=classes)
    predicted = np.bincount(predictions.ravel(), minlength=classes)
    support = np.bincount(truth, minlength=classes)
    if np.any(support == 0):
        raise ValueError("Each benchmark class needs at least one labeled example.")
    precision = np.divide(true_positives, predicted, out=np.zeros(classes), where=predicted > 0)
    return {
        "correct": int(matches.sum()),
        "recall": round(float(np.mean(true_positives / support)), 9),
        "precision": round(float(precision.mean()), 9),
    }


def essentia_scores(dataset: str, track_ids, expected_md5) -> np.ndarray:
    """Read the verified test profiles for the same pinned audio recordings."""
    rows = []
    for track_id in track_ids:
        result = json.loads((CACHE / dataset / f"{track_id}.json").read_text())
        scores = np.asarray(result["scores"], dtype=float)
        if (result["version"] != ESSENTIA_VERSION or result["md5"] != expected_md5[track_id]
                or scores.shape != (len(KEY_NAMES),) or not np.isfinite(scores).all()):
            raise ValueError(f"Invalid Essentia test checkpoint for {dataset} {track_id}.")
        rows.append(scores)
    return np.stack(rows)


def main() -> None:
    """Score the frozen engines and check that all hit counts match the earlier report."""
    benches = benchmark(ROOT)
    giantsteps = benches["giantsteps"]
    checksum_dir = Path("data/annotations/giantsteps-key-dataset/md5")
    checksums = {
        "giantsteps": {
            track_id: (checksum_dir / f"{track_id}.LOFI.md5").read_text().split()[0].lower()
            for track_id in giantsteps.loc[giantsteps.split == "test", "track_id"]
        },
        "fmak": pd.read_parquet(FMA_ROOT / "features.parquet").set_index("track_id")["md5"],
    }
    model = SoftVotingKeyModel.load(PUBLIC / "soft_vote_calibration.json")
    scorers = open_feature_scorers()
    targets, scores_by_engine, sizes = [], {engine: [] for engine in METHODS}, {}
    with threadpool_limits(limits=2):
        for dataset, bench in benches.items():
            rows = bench.loc[bench.split == "test"]
            features, truth = xy(rows)
            scores = {engine: predict(features) for engine, predict in scorers.items()}
            scores["essentia"] = essentia_scores(dataset, rows.track_id, checksums[dataset])
            scores["ensemble"] = model.scores({engine: scores[engine] for engine in ENGINES})
            for engine, values in scores.items():
                scores_by_engine[engine].append(values)
            targets.append(truth)
            sizes[dataset] = len(truth)
    truth = np.concatenate(targets)
    summary = pd.read_csv("reports/model-diagnostics/ensemble-comparison.csv")
    prior = summary.loc[summary.dataset.eq("combined")].set_index("engine")
    methods = {}
    for engine, (method, report_name) in METHODS.items():
        scores = np.concatenate(scores_by_engine[engine])
        exact = np.argsort(-scores, axis=1, kind="stable")[:, :3]
        pair_top3 = np.array([
            [PAIR_BY_KEY[first] for first, _, _ in rank_relative_pairs(row)[:3]]
            for row in scores
        ])
        results = {
            "first_exact": clip_metrics(truth, exact[:, :1], len(KEY_NAMES)),
            "top3_exact": clip_metrics(truth, exact, len(KEY_NAMES)),
            "top3_pairs": clip_metrics(PAIR_BY_KEY[truth], pair_top3, len(KEY_NAMES) // 2),
        }
        expected = prior.loc[report_name, ["exact_first_hits", "exact_top3_hits", "relative_pairs_top3_hits"]]
        if [row["correct"] for row in results.values()] != expected.tolist():
            raise ValueError(f"Frozen benchmark counts changed for {report_name}.")
        methods[method] = results

    aggregate = {"clips": len(truth),
                 "datasets": {"FMAK": sizes["fmak"], "GiantSteps": sizes["giantsteps"]},
                 "methods": methods}
    atomic_json(PUBLIC / "method_benchmarks.json", aggregate)


if __name__ == "__main__":
    main()
