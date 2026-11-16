# SPDX-License-Identifier: AGPL-3.0-only
"""Calibrate the five public key engines and combine their soft votes."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax

from .cloud_audio import SAMPLE_RATE
from .cloud_engines import essentia_scores
from .keys import KEY_NAMES
from .open_candidate import CandidateKeyModel
from .open_linear_svm import LinearKeySVM
from .open_tree_models import load_tree_models
from .research_features import release_histograms

ENGINES = ("current", "forest", "boosting", "linear", "essentia")
PROBABILITY_ENGINES = frozenset(("current", "forest", "boosting"))
FORMAT = "open-soft-voting-v1"


@lru_cache(maxsize=1)
def open_feature_scorers() -> dict:
    """Load the public engines that use the shared release chroma features."""
    trees = load_tree_models()
    neural = CandidateKeyModel.load(Path(__file__).with_name("open_key_model.json"))
    return {"current": neural.scores, "forest": trees["forest"].predict_proba,
            "boosting": trees["boosting"].predict_proba, "linear": LinearKeySVM.load().scores}


def _logits(scores: dict[str, np.ndarray]) -> np.ndarray:
    """Keep raw match scores and margins, and log-transform probabilities."""
    if set(scores) != set(ENGINES):
        raise ValueError("Soft voting requires scores from each of the five public engines.")
    scores = np.stack([scores[engine] for engine in ENGINES]).astype(float, copy=False)
    if scores.ndim != 3 or scores.shape[-1] != len(KEY_NAMES) or not np.isfinite(scores).all():
        raise ValueError("Scores must be finite with 24 keys per clip.")
    for index, engine in enumerate(ENGINES):
        if engine in PROBABILITY_ENGINES:
            probabilities = scores[index]
            if np.any(probabilities < 0) or not np.allclose(probabilities.sum(axis=1), 1):
                raise ValueError(f"Invalid class probabilities from {engine}.")
            scores[index] = np.log(np.maximum(probabilities, 1e-12))
    return scores


@dataclass(frozen=True)
class SoftVotingKeyModel:
    """Average one normalized, temperature-scaled vote per public engine."""

    temperatures: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.temperatures) != len(ENGINES) or not np.all(np.isfinite(self.temperatures)):
            raise ValueError("A finite temperature is required for each engine.")
        if min(self.temperatures) <= 0:
            raise ValueError("Temperatures must be positive.")

    @classmethod
    def fit(cls, scores: dict[str, np.ndarray], targets: np.ndarray,
            sample_weight: np.ndarray) -> SoftVotingKeyModel:
        """Fit one scalar per engine by minimizing validation log loss."""
        matrices = _logits(scores)
        targets = np.asarray(targets)
        weights = np.asarray(sample_weight, dtype=float)
        if (targets.shape != (matrices.shape[1],) or
                not np.issubdtype(targets.dtype, np.integer) or
                not len(targets) or np.any((targets < 0) | (targets >= len(KEY_NAMES))) or
                weights.shape != targets.shape or not np.isfinite(weights).all() or np.any(weights <= 0)):
            raise ValueError("Calibration needs one key label and positive weight per clip.")

        temperatures = []
        rows = np.arange(len(targets))
        for logits in matrices:
            def loss(log_temperature: float) -> float:
                scaled = logits * np.exp(-log_temperature)
                terms = logsumexp(scaled, axis=1) - scaled[rows, targets]
                return float(np.average(terms, weights=weights))

            # Include the default to retain it when scores supply no calibration gain.
            result = minimize_scalar(loss, bounds=(-np.log(100), np.log(100)), method="bounded")
            if not result.success:
                raise RuntimeError("Temperature calibration failed.")
            temperatures.append(float(np.exp(result.x)) if result.fun < loss(0) - 1e-9 else 1.0)
        return cls(tuple(temperatures))

    @classmethod
    def load(cls, path: Path) -> SoftVotingKeyModel:
        """Read a saved calibration file and verify its key and engine orders."""
        data = json.loads(path.read_text())
        if (data["format"] != FORMAT or tuple(data["classes"]) != KEY_NAMES
                or tuple(data["engines"]) != ENGINES):
            raise ValueError("Unsupported soft-voting calibration file.")
        return cls(tuple(data["temperatures"]))

    def distributions(self, scores: dict[str, np.ndarray]) -> np.ndarray:
        """Return normalized votes in engine, clip, key order."""
        return softmax(_logits(scores) / np.asarray(self.temperatures)[:, None, None], axis=-1)

    def scores(self, scores: dict[str, np.ndarray]) -> np.ndarray:
        """Return the equal-weight average over the five votes for each key."""
        return self.distributions(scores).mean(axis=0)

    def score_audio(self, samples: np.ndarray, *, feature_scorers: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return chroma and the combined key scores for decoded mono audio."""
        blocks = release_histograms(samples, SAMPLE_RATE)
        if np.ptp(blocks[0]) < 1e-8:
            raise ValueError("No usable pitch profile was detected.")
        features = blocks.reshape(1, -1)
        scorers = open_feature_scorers() if feature_scorers is None else feature_scorers
        scores = {name: score(features) for name, score in scorers.items()}
        _, essentia = essentia_scores(samples)
        scores["essentia"] = essentia.reshape(1, -1)
        return blocks[0], self.scores(scores)[0]
