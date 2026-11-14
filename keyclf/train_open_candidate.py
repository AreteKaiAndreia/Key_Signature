"""Train an equivariant per-key network and verify the frozen release candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

from .fmak import atomic_json
from .fmak_experiment import file_hash
from .keys import KEY_NAMES
from .open_candidate import FORMAT, CandidateKeyModel, candidate_features
from .open_harmonic_model import benchmark, metrics, synthetic_blocks, xy
from .open_loops import ROOT
from .train_open_model import score

OUT = Path("reports/open-key-candidate")


def run(work: Path, out: Path, final: bool) -> None:
    loops_path = work / "freesound-extended.parquet"
    benches = benchmark(work)
    artifact, receipt = out / "open-key-model.json", out / "selection.json"
    if not receipt.exists():
        loops = pd.read_parquet(loops_path)
        lx, ly = xy(loops)
        sx, sy = synthetic_blocks(work)
        # The original real-audio test rows are excluded from model selection.
        val = {"giantsteps": xy(benches["giantsteps"].loc[benches["giantsteps"].split != "test"]),
               "fmak": xy(benches["fmak"].loc[benches["fmak"].split == "val"])}
        candidates = {}
        selected = None
        for corpus, x, y in (("loops", lx, ly), ("combined", np.vstack((lx, sx)), np.r_[ly, sy])):
            pairs = candidate_features(x)
            target = y[:, None] == np.arange(24)
            distances = (np.arange(24)[None, :] - y[:, None]) % 12
            # Preserve all competing keys while emphasizing relative and parallel confusions.
            weights = np.where(target, 10., np.where(np.isin(distances, (0, 3, 9)), 2., 1.))
            for size in (24, 48):
                for regularization in (.03, .3):
                    estimator = MLPClassifier(hidden_layer_sizes=(size,), learning_rate_init=.004,
                                              batch_size=192, alpha=regularization, max_iter=72, random_state=42)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", ConvergenceWarning)
                        estimator.fit(pairs.reshape(-1, 37), target.ravel(), sample_weight=weights.ravel())
                    for offset in (0., .35, .7):
                        model = CandidateKeyModel(estimator.coefs_[0], estimator.intercepts_[0],
                                                  estimator.coefs_[1][:, 0], estimator.intercepts_[1][0], offset)
                        name = f"{corpus}, hidden={size}, alpha={regularization:g}, minor={offset:g}"
                        result = {label: metrics(model, *rows) for label, rows in val.items()}
                        candidates[name] = result
                        if selected is None or score(result) > score(candidates[selected]):
                            selected, best = name, model
                    print(f"Fitted {corpus} with {size} units and alpha {regularization:g}", flush=True)
        value = {"format": FORMAT, "classes": list(KEY_NAMES), "hidden": best.hidden.tolist(),
                 "hidden_bias": best.hidden_bias.tolist(), "output": best.output.tolist(),
                 "output_bias": best.output_bias, "minor_offset": best.minor_offset, "variant": selected}
        atomic_json(artifact, value)
        atomic_json(receipt, {"selected": selected, "validation": candidates, "model_sha256": file_hash(artifact),
                              "training_sha256": file_hash(loops_path)})
    choice = json.loads(receipt.read_text())
    if choice["model_sha256"] != file_hash(artifact) or choice["training_sha256"] != file_hash(loops_path):
        raise ValueError("Frozen candidate or training data changed.")
    print(json.dumps({"selected": choice["selected"], "validation": choice["validation"][choice["selected"]]}, indent=2))
    if final:
        model = CandidateKeyModel.load(artifact)
        results = {name: metrics(model, *xy(frame.loc[frame.split == "test"])) for name, frame in benches.items()}
        outcome = {"passed": all(r["first_rate"] >= .5 and r["top3_rate"] >= .7 for r in results.values()),
                   "test": results, "model_sha256": choice["model_sha256"]}
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
