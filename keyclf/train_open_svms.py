"""Fit SVM challengers on the open corpus and score the frozen real-audio splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from .evaluate_relative_pairs import measure
from .experiments import rotate_training
from .fmak import atomic_json
from .fmak_experiment import file_hash
from .keys import KEY_NAMES
from .open_candidate import CandidateKeyModel
from .open_harmonic_model import GS_RESEARCH, benchmark, synthetic_blocks, xy
from .open_linear_svm import ARTIFACT, FORMAT, LinearKeySVM
from .open_loops import ROOT
from .train_open_candidate import OUT as LIVE_REPORTS
from .train_open_model import SYNTH_COUNT

OUT = Path("reports/open-key-svm")
C_VALUES = (0.1, 1.0, 10.0)
GAMMAS = ("scale", 0.01, 0.1)
LIVE_MODEL = Path(__file__).with_name("open_key_model.json")


def save_estimator(path: Path, estimator) -> None:
    """Replace a fitted model only after its serialization has finished."""
    with NamedTemporaryFile(dir=path.parent, suffix=".joblib", delete=False) as file:
        temporary = Path(file.name)
        try:
            joblib.dump(estimator, file, compress=3)
            file.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def rank_validation(results: dict) -> tuple[float, float]:
    """Weight the two real-audio development sets equally."""
    return tuple(
        sum(result["exact"][cutoff] / result["n"] for result in results.values()) / len(results)
        for cutoff in ("first", "top3")
    )


def select(work: Path, out: Path, benches: dict, cache_hashes: dict) -> dict:
    """Fit each kernel and write the two chosen estimators and validation evidence."""
    loops_x, loops_y = xy(pd.read_parquet(work / "freesound-extended.parquet"))
    synth_x, synth_y = synthetic_blocks(work)
    x, y = rotate_training(np.vstack((loops_x, synth_x)), np.r_[loops_y, synth_y])
    validation = {
        "giantsteps": xy(benches["giantsteps"].loc[benches["giantsteps"].split != "test"]),
        "fmak": xy(benches["fmak"].loc[benches["fmak"].split == "val"]),
    }
    selected = {}
    for kernel in ("linear", "rbf"):
        candidates = []
        best = None
        for c in C_VALUES:
            for gamma in GAMMAS if kernel == "rbf" else (None,):
                params = {"C": c} if gamma is None else {"C": c, "gamma": gamma}
                estimator = make_pipeline(StandardScaler(), SVC(
                    kernel=kernel, decision_function_shape="ovr", break_ties=True, cache_size=256, **params,
                ))
                estimator.fit(x, y)
                if not np.array_equal(estimator.classes_, np.arange(24)):
                    raise ValueError("The SVM classes do not match the app's key order.")
                results = {name: measure(estimator.decision_function(features), keys)
                           for name, (features, keys) in validation.items()}
                rank = rank_validation(results)
                trial = {"params": params, "validation": results, "rank": list(rank)}
                candidates.append(trial)
                if best is None or rank > best[0]:
                    best = rank, trial, estimator
                print(f"{kernel} {params}: validation exact {rank[0]:.1%} first, {rank[1]:.1%} top three", flush=True)
        _, winner, estimator = best
        artifact = out / f"{kernel}.joblib"
        save_estimator(artifact, estimator)
        selected[kernel] = {"selected": winner, "model_sha256": file_hash(artifact), "trials": candidates}
    result = {"training_rows": {"freesound": len(loops_x), "synthetic": len(synth_x), "augmented": len(y)},
              "inputs": cache_hashes, "models": selected}
    atomic_json(out / "selection.json", result)
    return result


def load_selected(out: Path, selection: dict, kernel: str):
    """Load the local model with the hash recorded at selection."""
    artifact = out / f"{kernel}.joblib"
    if file_hash(artifact) != selection["models"][kernel]["model_sha256"]:
        raise ValueError(f"The selected {kernel} SVM changed after validation.")
    return joblib.load(artifact)


def export_linear(out: Path, benches: dict, selection: dict) -> None:
    """Export linear coefficients after checking parity with the frozen estimator."""
    pipeline = load_selected(out, selection, "linear")
    scaler, svc = pipeline.steps[0][1], pipeline.steps[1][1]
    if svc.kernel != "linear" or not np.array_equal(svc.classes_, np.arange(len(KEY_NAMES))):
        raise ValueError("Expected the selected linear SVM in the app's key order.")
    weights = svc.coef_ / scaler.scale_
    intercepts = svc.intercept_ - weights @ scaler.mean_
    compact = LinearKeySVM(weights, intercepts)
    report = json.loads((out / "test.json").read_text())
    if report["selection_sha256"] != file_hash(out / "selection.json"):
        raise ValueError("The test report belongs to a different SVM selection.")
    for name, frame in benches.items():
        features, keys = xy(frame)
        expected, actual = pipeline.decision_function(features), compact.scores(features)
        np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=0)
        np.testing.assert_array_equal(np.argsort(-actual, axis=1, kind="stable"),
                                      np.argsort(-expected, axis=1, kind="stable"))
        mask = frame.split == "test"
        if measure(actual[mask], keys[mask]) != report["test"][name]["linear"]:
            raise ValueError(f"The exported linear SVM changed the held-out results on {name}.")
        print(f"{name}: identical key order on {len(frame)} clips and matching held-out counts.", flush=True)
    atomic_json(ARTIFACT, {"format": FORMAT, "classes": list(KEY_NAMES),
                           "weights": weights.tolist(), "intercepts": intercepts.tolist()})
    print(f"{ARTIFACT.name}: SHA-256 {file_hash(ARTIFACT)}")


def evaluate(out: Path, benches: dict, selection: dict, live_receipt: dict) -> dict:
    """Score only the frozen winners and verify the live model's prior results."""
    scorers = {"current": CandidateKeyModel.load(LIVE_MODEL).scores}
    for kernel in selection["models"]:
        scorers[kernel] = load_selected(out, selection, kernel).decision_function
    results = {}
    for name, frame in benches.items():
        features, keys = xy(frame.loc[frame.split == "test"])
        results[name] = {model: measure(predict(features), keys) for model, predict in scorers.items()}
        if results[name]["current"] != live_receipt["test"][name]:
            raise ValueError(f"The frozen app results no longer match on {name}.")
    final = {"selection_sha256": file_hash(out / "selection.json"),
             "current_model_sha256": live_receipt["model_sha256"], "test": results}
    atomic_json(out / "test.json", final)
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=OUT)
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--final", action="store_true", help="Evaluate already-selected models on the test sets.")
    operation.add_argument("--export-linear", action="store_true", help="Verify and export the scored linear winner.")
    args = parser.parse_args()

    work, out = args.work, args.out
    loops = work / "freesound-extended.parquet"
    live_selection = json.loads((LIVE_REPORTS / "selection.json").read_text())
    live_receipt = json.loads((LIVE_REPORTS / "relative-pairs.json").read_text())
    if file_hash(loops) != live_selection["training_sha256"] or file_hash(LIVE_MODEL) != live_receipt["model_sha256"]:
        raise ValueError("The open training cache or live model does not match its frozen receipt.")
    paths = [loops, GS_RESEARCH, work / "fmak-bench.parquet"]
    paths += [work / "synthetic-extended" / f"{index}.json" for index in range(SYNTH_COUNT)]
    cache_hashes = {str(path): file_hash(path) for path in paths}
    selection_path = out / "selection.json"
    with threadpool_limits(limits=2):
        benches = benchmark(work)
        if selection_path.exists():
            selection = json.loads(selection_path.read_text())
            if selection["inputs"] != cache_hashes:
                raise ValueError("The training or benchmark cache changed after validation.")
        elif args.final or args.export_linear:
            parser.error("Run model selection without --final first.")
        else:
            out.mkdir(parents=True, exist_ok=True)
            selection = select(work, out, benches, cache_hashes)
        if args.export_linear:
            export_linear(out, benches, selection)
        elif args.final:
            print(json.dumps(evaluate(out, benches, selection, live_receipt), indent=2))
        else:
            print(json.dumps({name: receipt["selected"] for name, receipt in selection["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
