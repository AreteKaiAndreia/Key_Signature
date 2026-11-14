"""Matched feature ablation and frozen, real-audio Hooktheory expansion tests."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .experiments import estimator_for, rotate_training
from .features import feature_names
from .fmak import HISTOGRAM, atomic_json, digest, load_manifest, validate_checkpoint
from .fmak_experiment import COUNTS, file_hash, save_model, split_ids
from .hooktheory import FMA_CACHE, ROOT as HOOK_ROOT
from .keys import KEY_NAMES, parse_key
from .train import probabilities_24

OUTPUT = Path("reports/hooktheory")
BASE = Path("data/features/giantsteps_features.parquet")
REFERENCE = Path("models/research_model.joblib")
CLOUD_MODEL = Path("models/cloud_model.joblib")
VERSION = "real-song-hist-ablation-hook-v1"
CONFIG = {"algorithm": "forest", "leaf": 2, "max_features": "sqrt"}
WEIGHTS = (.25, .5, 1.0)


def ranked_keys(scores: np.ndarray) -> np.ndarray:
    """Rank canonical keys by descending score, breaking ties by class ID."""
    return np.argsort(-np.asarray(scores), axis=-1, kind="stable")


def measure(model, frame: pd.DataFrame, columns: list[str]) -> dict:
    """Measure first-choice and top-three accuracy with explicit counts."""
    y = np.asarray([parse_key(k) for k in frame.key])
    with threadpool_limits(limits=2):
        p = probabilities_24(model, frame[columns].to_numpy(dtype=float))
    order = ranked_keys(p)
    correct = order[:, 0] == y
    top3 = (order[:, :3] == y[:, None]).any(axis=1)
    return {"n": len(y), "top1": int(correct.sum()), "top3": int(top3.sum()),
            "top1_rate": float(correct.mean()), "top3_rate": float(top3.mean()),
            "log_loss": float(-np.log(np.maximum(np.finfo(float).eps, p[np.arange(len(y)), y])).mean())}


def fit(rows: list[tuple[pd.DataFrame, float]], columns: list[str]):
    """Fit the same seeded forest on original rows plus twelve transpositions."""
    frame = pd.concat([frame for frame, _ in rows], ignore_index=True)
    y = np.array([parse_key(k) for k in frame.key])
    x, y = rotate_training(frame[columns].to_numpy(dtype=float), y)
    weight = np.tile(np.concatenate([np.full(len(df), w) for df, w in rows]), 12)
    model = estimator_for(CONFIG)
    with threadpool_limits(limits=2):
        model.fit(x, y, sample_weight=weight if (weight != 1).any() else None)
    return model


def validate_inputs(base, fma, hook, ref: dict, fma_path: Path, hook_work: Path) -> dict:
    """Ensure the same held-out rows and no added nontraining Hooktheory rows."""
    ids = {"giantsteps": split_ids(base), "fmak": split_ids(fma)}
    if ids["giantsteps"] != ref["splits"] or {k: len(v) for k, v in ids["giantsteps"].items()} != COUNTS:
        raise ValueError("GiantSteps recording partitions changed")
    if {k: len(v) for k, v in ids["fmak"].items()} != {"train": 350, "val": 75, "test": 75}:
        raise ValueError("expected the frozen 500-song FMAK pilot")
    manifest = load_manifest(fma_path.parent)
    planned = {row["track_id"]: row for row in manifest["tracks"]}
    if set(fma.track_id) != set(planned) or not all(
        validate_checkpoint(row, planned[row["track_id"]], manifest["signature"])
        for row in fma.to_dict("records")
    ):
        raise ValueError("FMAK rows differ from the frozen manifest")
    m = json.loads((hook_work / "manifest.json").read_text())
    hids = sorted(hook.track_id)
    if m["signature"] != digest({k: v for k, v in m.items() if k != "signature"}) or (
        not 1 <= len(hids) <= 500 or hook.track_id.duplicated().any() or set(hook.split) != {"train"}
        or hids != sorted(row["track_id"] for row in m["tracks"]) or set(hook.manifest) != {m["signature"]}
    ):
        raise ValueError("invalid Hooktheory training-only manifest or cache")
    labels = {r["track_id"]: r["key"] for r in m["tracks"]}
    if any(labels[i] != k for i, k in zip(hook.track_id, hook.key, strict=True)):
        raise ValueError("Hooktheory labels differ from manifest")
    if len(set(base.track_id) | set(fma.track_id) | set(hook.track_id)) != len(base) + len(fma) + len(hook):
        raise ValueError("recording overlap")
    for frame in (base, fma, hook):
        p = frame[HISTOGRAM].to_numpy(dtype=float)
        if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1):
            raise ValueError("invalid chroma histogram")
    return ids


def rank(scores: dict) -> tuple:
    """Rank on mean real-audio validation scores, with no test inputs."""
    return (sum(v["top1_rate"] for v in scores.values()),
            sum(v["top3_rate"] for v in scores.values()),
            -sum(v["log_loss"] for v in scores.values()))


def choose(validation: dict[str, dict]) -> tuple[str, str]:
    """Select a Hooktheory weight and apply the GiantSteps default guard."""
    base = validation["chroma"]
    hook = max((f"hook-{w:g}" for w in WEIGHTS), key=lambda k: rank(validation[k]))
    eligible = [k for k in ("fmak", hook)
                if rank(validation[k])[0] > rank(base)[0]
                and validation[k]["giantsteps"]["top1"] >= base["giantsteps"]["top1"]]
    winner = max(eligible, key=lambda k: rank(validation[k])) if eligible else "chroma"
    return hook, winner


def run(base_path: Path, fma_path: Path, hook_work: Path, reference_path: Path,
        output: Path, cloud_model: Path) -> dict:
    """Freeze all selection decisions before evaluating the two real test partitions."""
    hook_path = hook_work / "features.parquet"
    inputs = {k: file_hash(path) for k, path in
              (("base", base_path), ("fmak", fma_path), ("hook", hook_path), ("reference", reference_path))}
    signature = digest({"version": VERSION, "inputs": inputs, "weights": WEIGHTS, "config": CONFIG})
    selection_file, receipt_file = output / "selection.json", output / "results.json"
    if receipt_file.exists():
        receipt = json.loads(receipt_file.read_text())
        if receipt["signature"] != signature or receipt["cloud_weights_sha256"] != file_hash(cloud_model):
            raise ValueError("preserve the previous test receipt and its matching model")
        return receipt
    if len({base_path.resolve(), fma_path.resolve(), hook_path.resolve(), reference_path.resolve(), cloud_model.resolve()}) != 5:
        raise ValueError("the deployment artifact must not overwrite any inputs")
    base, fma, hook = [pd.read_parquet(p).sort_values("track_id") for p in (base_path, fma_path, hook_path)]
    reference = joblib.load(reference_path)
    if reference["feature_names"] != HISTOGRAM or tuple(reference["classes"]) != KEY_NAMES:
        raise ValueError("reference is not the existing chroma-only forest")
    reference_model = reference["model"]
    ids = validate_inputs(base, fma, hook, reference, fma_path, hook_work)
    variants = {"edges": (feature_names(), False, 0), "chroma": (HISTOGRAM, False, 0),
                "fmak": (HISTOGRAM, True, 0),
                **{f"hook-{w:g}": (HISTOGRAM, True, w) for w in WEIGHTS}}

    def fit_variant(name: str, final: bool):
        cols, add_fma, w = variants[name]
        rows = [(base[base.split != "test"] if final else base[base.split == "train"], 1.)]
        if add_fma:
            rows.append((fma[fma.split != "test"] if final else fma[fma.split == "train"], 1.))
        if w:
            rows.append((hook, w))
        return fit(rows, cols), cols, [len(df) for df, _ in rows]

    if selection_file.exists():
        selection = json.loads(selection_file.read_text())
        if selection["signature"] != signature:
            raise ValueError("use a fresh output directory rather than modifying frozen selection")
    else:
        validation = {}
        for name in variants:
            model, cols, _ = fit_variant(name, final=False)
            sources = {"giantsteps": base}
            if name != "edges":  # FMAK cached only histogram columns.
                sources["fmak"] = fma
            validation[name] = {k: measure(model, df[df.split == "val"], cols) for k, df in sources.items()}
            print("validation", name, {k: f"{v['top1']}/{v['n']}" for k, v in validation[name].items()}, flush=True)
            del model
            gc.collect()
        hook_winner, winner = choose(validation)
        selection = {"signature": signature, "inputs": inputs, "validation": validation,
                     "test_ids_sha256": {k: digest({"ids": v["test"]}) for k, v in ids.items()},
                     "hook_candidate": hook_winner, "cloud_candidate": winner}
        atomic_json(selection_file, selection)
    print("Selection frozen:", selection["hook_candidate"], "cloud:", selection["cloud_candidate"], flush=True)
    results = {}
    for name in ("edges", "chroma", "fmak", selection["hook_candidate"]):
        model, cols, counts = fit_variant(name, final=True)
        sources = {"giantsteps": base}
        if name != "edges":
            sources["fmak"] = fma
        results[name] = {"train_n": sum(counts),
                         "test": {k: measure(model, df[df.split == "test"], cols) for k, df in sources.items()}}
        if name == selection["cloud_candidate"]:
            chosen = reference_model if name == "chroma" else model
            save_model(cloud_model, {"model": chosen, "feature_names": HISTOGRAM,
                                    "classes": list(KEY_NAMES), "variant": name, "version": VERSION,
                                    "experiment_signature": signature})
        del model
        gc.collect()
        print("test", name, results[name]["test"], flush=True)
    reference_scores = {k: measure(reference_model, frame[frame.split == "test"], HISTOGRAM)
                        for k, frame in {"giantsteps": base, "fmak": fma}.items()}
    changes = {}
    for name, reference_name in (("chroma", "edges"), ("fmak", "chroma"), (selection["hook_candidate"], "fmak")):
        changes[name] = {source: {f"{metric}_delta": score[metric] - results[reference_name]["test"][source][metric]
                                    for metric in ("top1", "top3", "top1_rate", "top3_rate")}
                         for source, score in results[name]["test"].items() if source in results[reference_name]["test"]}
    result = {"signature": signature, "selection": selection, "test": results, "changes": changes,
              "frozen_reference": reference_scores, "cloud_weights_sha256": file_hash(cloud_model),
              "test_used_for_selection": False, "rendered_examples_in_test": 0}
    atomic_json(receipt_file, result)
    return result


def main() -> None:
    """Execute or replay the complete matched experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--fma", type=Path, default=FMA_CACHE)
    parser.add_argument("--hook", type=Path, default=HOOK_ROOT)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--cloud-model", type=Path, default=CLOUD_MODEL)
    opts = parser.parse_args()
    result = run(opts.base, opts.fma, opts.hook, opts.reference, opts.output, opts.cloud_model)
    print(json.dumps({"cloud_candidate": result["selection"]["cloud_candidate"], "test": result["test"]}, indent=2))


if __name__ == "__main__":
    main()
