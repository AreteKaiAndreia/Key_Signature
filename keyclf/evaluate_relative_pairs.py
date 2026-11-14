"""Compare exact-key and relative-pair accuracy on the frozen test sets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .fmak import atomic_json
from .fmak_experiment import file_hash
from .open_candidate import CandidateKeyModel
from .open_harmonic_model import benchmark, xy
from .open_loops import ROOT
from .relative_pairs import rank_relative_pairs
from .train_open_candidate import OUT


def measure(scores: np.ndarray, targets: np.ndarray) -> dict:
    """Count hits in the first guess and the first three displayed guesses."""
    exact = np.argsort(-scores, axis=1, kind="stable")[:, :3] == targets[:, None]
    grouped = np.array([
        [target in (key, relative) for key, relative, _ in rank_relative_pairs(row)[:3]]
        for row, target in zip(scores, targets, strict=True)
    ])
    return {
        "n": len(targets),
        "exact": {"first": int(exact[:, 0].sum()), "top3": int(exact.any(axis=1).sum())},
        "relative_pairs": {"first": int(grouped[:, 0].sum()), "top3": int(grouped.any(axis=1).sum())},
    }


def main() -> None:
    """Verify the app's frozen model and write an aggregate-only report."""
    artifact = Path(__file__).with_name("open_key_model.json")
    prior = json.loads((OUT / "test.json").read_text())
    sha = file_hash(artifact)
    if sha != prior["model_sha256"]:
        raise ValueError("The app model differs from the frozen benchmark model.")
    model = CandidateKeyModel.load(artifact)
    results = {}
    with threadpool_limits(limits=2):
        for name, frame in benchmark(ROOT).items():
            features, targets = xy(frame.loc[frame.split == "test"])
            result = measure(model.scores(features), targets)
            previous = prior["test"][name]
            if {"n": result["n"], **result["exact"]} != {key: previous[key] for key in ("n", "first", "top3")}:
                raise ValueError(f"Exact-key accuracy no longer matches the frozen result for {name}.")
            results[name] = result
    report = {"model_sha256": sha, "test": results}
    atomic_json(OUT / "relative-pairs.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
