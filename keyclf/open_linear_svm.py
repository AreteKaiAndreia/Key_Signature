"""Run the selected linear SVM from pairwise coefficients."""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

from .cloud_audio import SAMPLE_RATE
from .keys import KEY_NAMES
from .research_features import release_histograms

ARTIFACT = Path(__file__).with_name("linear_svm.json")
FORMAT = "open-linear-svm-v1"
KEY_PAIRS = tuple(combinations(range(len(KEY_NAMES)), 2))


@dataclass(frozen=True)
class LinearKeySVM:
    """Score each key by pairwise votes with margin tie-breaks."""

    weights: np.ndarray
    intercepts: np.ndarray

    @classmethod
    def load(cls, path: Path = ARTIFACT) -> LinearKeySVM:
        """Load the published coefficients in the app's key order."""
        value = json.loads(path.read_text())
        if value["format"] != FORMAT or tuple(value["classes"]) != KEY_NAMES:
            raise ValueError("Unsupported linear SVM artifact.")
        weights = np.asarray(value["weights"], dtype=float)
        intercepts = np.asarray(value["intercepts"], dtype=float)
        if (weights.shape != (len(KEY_PAIRS), 36) or intercepts.shape != (len(KEY_PAIRS),)
                or not np.isfinite(weights).all() or not np.isfinite(intercepts).all()):
            raise ValueError("Invalid linear SVM coefficients.")
        return cls(weights, intercepts)

    def scores(self, features: np.ndarray) -> np.ndarray:
        """Reproduce the SVC multiclass decision scores from linear margins."""
        margins = np.asarray(features, dtype=float).reshape(-1, 36) @ self.weights.T
        margins += self.intercepts
        votes = np.zeros((len(margins), len(KEY_NAMES)))
        totals = np.zeros_like(votes)
        for column, (first, second) in enumerate(KEY_PAIRS):
            margin = margins[:, column]
            first_wins = margin >= 0
            votes[:, first] += first_wins
            votes[:, second] += ~first_wins
            totals[:, first] += margin
            totals[:, second] -= margin
        # A margin adjustment is smaller than a vote, so it only breaks ties.
        return votes + totals / (3 * (np.abs(totals) + 1))

    def score_audio(self, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Extract the three release chroma blocks and rank the keys."""
        features = release_histograms(samples, SAMPLE_RATE)
        if np.ptp(features[0]) < 1e-8:
            raise ValueError("No usable pitch profile was detected.")
        return features[0], self.scores(features)[0]
