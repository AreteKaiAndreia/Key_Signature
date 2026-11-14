"""Predict major/minor keys using a compact, rotation-equivariant classifier."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax

from .baseline import ks_scores
from .keys import KEY_NAMES

FORMAT = "open-chroma-key-v1"
POWERS = (0.5, 1.0, 2.0)


def transform(histograms: np.ndarray, power: float) -> np.ndarray:
    """Apply a fixed normalization before fitting or scoring."""
    rows = np.asarray(histograms, dtype=float)
    rows = np.power(rows, power)
    return rows / rows.sum(axis=-1, keepdims=True)


@dataclass(frozen=True)
class OpenKeyModel:
    """Two learned pitch profiles, rotated to score the 24 canonical keys."""

    profiles: np.ndarray
    mode_offsets: np.ndarray
    power: float
    ks_weight: float

    def predict_proba(self, histograms: np.ndarray) -> np.ndarray:
        rotated = np.vstack([np.roll(profile, tonic) for profile in self.profiles for tonic in range(12)])
        scores = transform(histograms, self.power) @ rotated.T + np.repeat(self.mode_offsets, 12)
        if self.ks_weight:
            scores += self.ks_weight * ks_scores(histograms)
        return softmax(scores, axis=-1)

    @classmethod
    def load(cls, path: Path) -> OpenKeyModel:
        """Load finite coefficients and a compatible class order from plain JSON."""
        data = json.loads(path.read_text())
        if data["format"] != FORMAT or tuple(data["classes"]) != KEY_NAMES:
            raise ValueError("Unsupported key model format or class order.")
        profiles = np.asarray(data["profiles"], dtype=float)
        offsets = np.asarray(data["mode_offsets"], dtype=float)
        power, weight = data["power"], data["ks_weight"]
        if (profiles.shape != (2, 12) or offsets.shape != (2,) or not np.isfinite(profiles).all()
                or not np.isfinite(offsets).all() or power not in POWERS or not 0 <= weight <= 10):
            raise ValueError("Invalid key model coefficients.")
        return cls(profiles, offsets, power, weight)
