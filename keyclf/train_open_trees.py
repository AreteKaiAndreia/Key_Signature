"""Train the public tree engines on the same open corpus as the neural model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from threadpoolctl import threadpool_limits

from .evaluate_relative_pairs import measure
from .experiments import rotate_training
from .fmak import atomic_json
from .fmak_experiment import file_hash
from .keys import KEY_NAMES
from .open_harmonic_model import benchmark, synthetic_blocks, xy
from .open_loops import ROOT
from .open_tree_models import ARTIFACT, FORMAT
from .train_open_candidate import OUT as LIVE_REPORTS
from .train_open_svms import rank_validation, save_estimator

OUT = Path("reports/open-key-trees")


def main() -> None:
    """Freeze the forest and boosting settings before scoring the test clips."""
    OUT.mkdir(parents=True, exist_ok=True)
    selection_path = OUT / "selection.json"
    loops_path = ROOT / "freesound-extended.parquet"
    training_sha = file_hash(loops_path)
    if training_sha != json.loads((LIVE_REPORTS / "selection.json").read_text())["training_sha256"]:
        raise ValueError("The Freesound cache differs from the frozen neural training corpus.")
    with threadpool_limits(limits=2):
        benches = benchmark(ROOT)
        if not selection_path.exists():
            loops, loop_keys = xy(pd.read_parquet(loops_path))
            synthetic, synth_keys = synthetic_blocks(ROOT)
            features, keys = rotate_training(np.vstack((loops, synthetic)), np.r_[loop_keys, synth_keys])
            development = {
                "giantsteps": xy(benches["giantsteps"].loc[benches["giantsteps"].split != "test"]),
                "fmak": xy(benches["fmak"].loc[benches["fmak"].split == "val"]),
            }
            candidates = {
                "forest": [
                    RandomForestClassifier(n_estimators=120, max_depth=depth, min_samples_leaf=4,
                                           random_state=42, n_jobs=2)
                    for depth in (13, 18)
                ],
                "boosting": [
                    HistGradientBoostingClassifier(max_iter=rounds, max_leaf_nodes=leaves,
                                                   early_stopping=False, random_state=42)
                    for rounds, leaves in ((60, 15), (90, 23))
                ],
            }
            models, validation = {}, {}
            for name, estimators in candidates.items():
                trials = []
                for estimator in estimators:
                    estimator.fit(features, keys)
                    if not np.array_equal(estimator.classes_, np.arange(len(KEY_NAMES))):
                        raise ValueError("The estimator classes differ from the app's key order.")
                    results = {dataset: measure(estimator.predict_proba(data), labels)
                               for dataset, (data, labels) in development.items()}
                    rank = rank_validation(results)
                    trials.append({"settings": estimator.get_params(), "validation": results, "rank": rank})
                    print(f"{name}: validation exact first {rank[0]:.1%}, top three {rank[1]:.1%}", flush=True)
                winner = max(range(len(trials)), key=lambda index: trials[index]["rank"])
                models[name] = estimators[winner]
                validation[name] = {"selected": winner, "trials": trials}
            models["forest"].n_jobs = 1
            save_estimator(ARTIFACT, {"format": FORMAT, "classes": KEY_NAMES, "models": models})
            atomic_json(selection_path, {"model_sha256": file_hash(ARTIFACT), "training_sha256": training_sha,
                                         "training_rows": len(keys), "validation": validation})

        selection = json.loads(selection_path.read_text())
        if selection["model_sha256"] != file_hash(ARTIFACT) or selection["training_sha256"] != training_sha:
            raise ValueError("The selected tree models or training data changed.")
        models = joblib.load(ARTIFACT)["models"]
        results = {}
        for name, frame in benches.items():
            data, labels = xy(frame.loc[frame.split == "test"])
            results[name] = {kind: measure(model.predict_proba(data), labels) for kind, model in models.items()}
        report = {"model_sha256": selection["model_sha256"], "test": results}
        atomic_json(OUT / "test.json", report)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
