"""Compare a frozen GiantSteps model with a disk-bounded FMAK expansion."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .experiments import estimator_for, rotate_training, summarize
from .fmak import HISTOGRAM, ROOT, atomic_json, digest, load_manifest, validate_checkpoint
from .keys import KEY_NAMES, parse_key
from .train import probabilities_24

VERSION = "fmak-domain-validation-1"
WEIGHTS = (.25, .5, 1.0)
SPLITS = ("train", "val", "test")
COUNTS = {"train": 422, "val": 91, "test": 91}


def file_hash(path: Path) -> str:
    """Hash a local artifact without loading it into a second memory buffer."""
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def split_ids(frame: pd.DataFrame) -> dict[str, list[str]]:
    """Return deterministic original recording IDs, not augmented rows."""
    if frame.track_id.isna().any() or frame.track_id.duplicated().any() or set(frame.split) != set(SPLITS):
        raise ValueError("features require unique recording IDs and all three partitions")
    return {split: sorted(frame.loc[frame.split == split, "track_id"].astype(str)) for split in SPLITS}


def validate_data(base: pd.DataFrame, new: pd.DataFrame, manifest: dict, reference: dict) -> dict:
    """Reject incomplete caches, modified original splits, and recording overlap."""
    old, fresh = split_ids(base), split_ids(new)
    if {split: len(ids) for split, ids in old.items()} != COUNTS or old != reference["splits"]:
        raise ValueError("GiantSteps rows or split IDs differ from the frozen model")
    if any(set(base.track_id.astype(str)) & set(ids) for ids in fresh.values()):
        raise ValueError("recording IDs cross data sources")
    tracks = {t["track_id"]: t for t in manifest["tracks"]}
    if set(new.track_id) != set(tracks) or new.md5.isna().any() or new.md5.duplicated().any():
        raise ValueError("FMAK cache is incomplete or has duplicate audio")
    if not all(validate_checkpoint(row, tracks[row["track_id"]], manifest["signature"])
               for row in new.to_dict("records")):
        raise ValueError("FMAK features do not match the frozen manifest")
    for frame in (base, new):
        values = frame[HISTOGRAM].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any() or not np.allclose(values.sum(axis=1), 1):
            raise ValueError("invalid normalized histogram")
    return {"giantsteps": old, "fmak": fresh}


def xy(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Use the same chroma histogram and canonical labels in both conditions."""
    return frame[HISTOGRAM].to_numpy(dtype=float), np.array([parse_key(k) for k in frame.key])


def fit(base: pd.DataFrame, new: pd.DataFrame, config: dict, weight: float):
    """Fit the fixed forest; only the added-source training weight varies."""
    x, y = xy(pd.concat([base, new], ignore_index=True))
    x, y = rotate_training(x, y)
    weights = np.tile(np.r_[np.ones(len(base)), np.full(len(new), weight)], 12)
    model = estimator_for(config)
    with threadpool_limits(limits=2):
        model.fit(x, y, sample_weight=weights)
    return model


def scores(model, frame: pd.DataFrame) -> tuple[dict, np.ndarray]:
    """Evaluate exact key, top-three, and recall on one named source split."""
    x, y = xy(frame)
    with threadpool_limits(limits=2):
        probability = probabilities_24(model, x)
    return summarize(y, probability), probability.argmax(axis=1) == y


def rank(item: dict) -> tuple:
    """Rank on equal-domain validation accuracy, top three, and log loss."""
    a, b = item["giantsteps"], item["fmak"]
    return (a["accuracy"] + b["accuracy"], a["top3_accuracy"] + b["top3_accuracy"],
            -(a["log_loss"] + b["log_loss"]))


def compare(reference, expanded, frame: pd.DataFrame) -> dict:
    """Report matched test metrics and paired exact-key wins and losses."""
    before, a = scores(reference, frame)
    after, b = scores(expanded, frame)
    return {"reference": before, "expanded": after,
            "exact_accuracy_delta": after["accuracy"] - before["accuracy"],
            "corrected": int((~a & b).sum()), "new_errors": int((a & ~b).sum())}


def save_model(path: Path, value: dict) -> None:
    """Atomically export the separate pilot model; never replace the reference."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".fmak-model-", delete=False) as file:
        pending = Path(file.name)
        try:
            joblib.dump(value, file, compress=3)
            file.close()
            os.replace(pending, path)
        finally:
            pending.unlink(missing_ok=True)


def run(work: Path, base_path: Path, reference_path: Path, output: Path, model_path: Path) -> dict:
    """Freeze selection on validation before evaluating both held-out sources."""
    if model_path.resolve() == reference_path.resolve():
        raise ValueError("the expanded model must not overwrite the frozen reference")
    manifest = load_manifest(work)
    new_path = work / "features.parquet"
    base = pd.read_parquet(base_path).sort_values("track_id")
    new = pd.read_parquet(new_path).sort_values("track_id")
    reference = joblib.load(reference_path)
    config = reference["config"]
    if reference["feature_names"] != HISTOGRAM or list(reference["classes"]) != list(KEY_NAMES) or config != {
        "name": "histogram_rf_transpose", "group": "histogram", "blocks": ["hist"],
        "algorithm": "forest", "augment": True, "leaf": 2, "max_features": "sqrt",
    }:
        raise ValueError("reference must be the frozen histogram research forest")
    ids = validate_data(base, new, manifest, reference)
    inputs = {"protocol": VERSION, "weights": list(WEIGHTS), "manifest": manifest["signature"],
              "original_features": file_hash(base_path), "fmak_features": file_hash(new_path),
              "reference_model": file_hash(reference_path), "model_config": config}
    signature = digest(inputs)
    selection_path, receipt = output / "selection.json", output / "test_results.json"
    if receipt.exists():
        prior = json.loads(receipt.read_text())
        if prior["signature"] != signature or prior["expanded_model_sha256"] != file_hash(model_path):
            raise ValueError("preserve the existing held-out receipt and matching model")
        return prior
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
        if selection["signature"] != signature:
            raise ValueError("selection was already frozen for different inputs")
    else:
        candidates = []
        for weight in WEIGHTS:
            model = fit(base[base.split == "train"], new[new.split == "train"], config, weight)
            candidate = {"fmak_weight": weight, "giantsteps": scores(model, base[base.split == "val"])[0],
                         "fmak": scores(model, new[new.split == "val"])[0]}
            candidates.append(candidate)
            print(f"weight={weight:g} validation: GiantSteps={candidate['giantsteps']['accuracy']:.1%}, "
                  f"FMAK={candidate['fmak']['accuracy']:.1%}", flush=True)
            del model
        selection = {"signature": signature, "inputs": inputs, "splits": ids, "candidates": candidates,
                     "ranking": "equal-domain validation exact accuracy, top-three, then lower log loss",
                     "selected_weight": max(candidates, key=rank)["fmak_weight"]}
        atomic_json(selection_path, selection)
    weight = selection["selected_weight"]
    if model_path.exists():
        artifact = joblib.load(model_path)
        if artifact.get("experiment_signature") != signature or artifact.get("fmak_weight") != weight:
            raise ValueError("refusing to replace an existing unrelated pilot model")
        model = artifact["model"]
    else:
        model = fit(base[base.split != "test"], new[new.split != "test"], config, weight)
        save_model(model_path, {"model": model, "feature_names": HISTOGRAM, "research_version": VERSION,
                                "experiment_signature": signature, "config": config, "classes": list(KEY_NAMES),
                                "fmak_weight": weight, "splits_by_source": ids})
    print("Selection frozen; evaluating the two fixed test sets.", flush=True)
    result = {"signature": signature, "inputs": inputs, "selected_weight": weight,
              "training_recordings": {source: len(parts["train"]) + len(parts["val"])
                                      for source, parts in ids.items()},
              "expanded_model_sha256": file_hash(model_path),
              "giantsteps_test": compare(reference["model"], model, base[base.split == "test"]),
              "fmak_test": compare(reference["model"], model, new[new.split == "test"]),
              "test_used_for_selection": False, "app_default_changed": False}
    atomic_json(receipt, result)
    return result


def main(argv: list[str] | None = None) -> int:
    """Train, evaluate, and report the opt-in expansion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=ROOT)
    parser.add_argument("--base-features", type=Path, default=Path("data/features/giantsteps_features.parquet"))
    parser.add_argument("--reference", type=Path, default=Path("models/research_model.joblib"))
    parser.add_argument("--output", type=Path, default=Path("reports/fmak-pilot"))
    parser.add_argument("--model", type=Path, default=Path("models/fmak_model.joblib"))
    args = parser.parse_args(argv)
    result = run(args.work, args.base_features, args.reference, args.output, args.model)
    for source in ("giantsteps_test", "fmak_test"):
        item = result[source]
        print(f"{source}: {item['reference']['accuracy']:.1%} -> {item['expanded']['accuracy']:.1%} "
              f"({item['corrected']} corrected, {item['new_errors']} new errors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
