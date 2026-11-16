"""Publish aggregate first-guess metrics from the local held-out engine comparison."""

from pathlib import Path
import json

import pandas as pd

REPORTS = Path("reports/model-diagnostics")
OUTPUT = Path("keyclf/method_benchmarks.json")
METHOD_IDS = {
    "Neural Network": "neural_network",
    "Linear SVM": "linear_svm",
    "Random Forest": "random_forest",
    "Gradient Boosting": "gradient_boosting",
    "Essentia": "essentia",
    "Soft voting (five engines)": "soft_voting",
}


def main() -> None:
    """Validate the shared test support and export accuracy, macro recall, and macro precision."""
    summary = pd.read_csv(REPORTS / "ensemble-comparison.csv")
    summary = summary.loc[summary.dataset.eq("combined")].set_index("engine")
    errors = pd.concat([
        pd.read_csv(REPORTS / "per-key-errors.csv"),
        pd.read_csv(REPORTS / "ensemble-per-key.csv"),
    ])
    errors = errors.loc[errors.dataset.eq("combined")]
    sizes = {"FMAK": 75, "GiantSteps": 91}
    total = sum(sizes.values())
    methods = {}
    support = None
    for name, method in METHOD_IDS.items():
        keys = errors.loc[errors.engine.eq(name)].set_index("key").sort_index()
        tp, fp, fn, tn = (keys[column] for column in
                          ("true_positives", "false_positives", "false_negatives", "true_negatives"))
        actual = tp + fn
        predicted = tp + fp
        correct = int(tp.sum())
        if (len(keys) != 24 or not keys.index.is_unique or actual.min() <= 0 or actual.sum() != total or predicted.sum() != total
                or not (actual + fp + tn).eq(total).all()
                or correct != summary.loc[name, "exact_first_hits"] or total != summary.loc[name, "n"]):
            raise ValueError(f"Inconsistent benchmark counts for {name}.")
        if support is not None and not support.equals(actual):
            raise ValueError(f"The true-key counts changed for {name}.")
        support = actual
        methods[method] = {
            "correct": correct,
            "recall": round(float((tp / actual).mean()), 9),
            "precision": round(float((tp / predicted.where(predicted > 0)).fillna(0).mean()), 9),
        }

    aggregate = {"clips": total, "datasets": sizes, "methods": methods}
    OUTPUT.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
