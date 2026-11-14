"""Score each candidate key against pitch evidence aligned to its tonic."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax

from .cloud_audio import SAMPLE_RATE
from .keys import KEY_NAMES
from .research_features import release_histograms

FORMAT = "open-candidate-key-v1"


def candidate_features(features: np.ndarray) -> np.ndarray:
    """Map a track to the same 37 inputs for every proposed key."""
    blocks = np.asarray(features, dtype=float).reshape(-1, 3, 12)
    blocks = np.sqrt(blocks)
    blocks /= np.maximum(blocks.sum(axis=-1, keepdims=True), np.finfo(float).eps)
    aligned = np.stack([np.roll(blocks, -tonic, axis=-1) for tonic in range(12)], axis=1)
    aligned = np.tile(aligned.reshape(-1, 12, 36), (1, 2, 1))
    mode = np.broadcast_to(np.repeat([0., 1.], 12), aligned.shape[:2])[..., None]
    return np.concatenate((5 * aligned, mode), axis=-1)


@dataclass(frozen=True)
class CandidateKeyModel:
    """Run a small neural classifier saved as numerical parameters."""

    hidden: np.ndarray
    hidden_bias: np.ndarray
    output: np.ndarray
    output_bias: float
    minor_offset: float

    def scores(self, features: np.ndarray) -> np.ndarray:
        logits = np.maximum(candidate_features(features) @ self.hidden + self.hidden_bias, 0) @ self.output
        logits[:, 12:] += self.minor_offset
        return softmax(logits + self.output_bias, axis=1)

    def score_audio(self, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        features = release_histograms(samples, SAMPLE_RATE)
        if np.ptp(features[0]) < 1e-8:
            raise ValueError("No usable pitch profile was detected.")
        return features[0], self.scores(features)[0]

    @classmethod
    def load(cls, path: Path) -> CandidateKeyModel:
        data = json.loads(path.read_text())
        if data["format"] != FORMAT or tuple(data["classes"]) != KEY_NAMES:
            raise ValueError("Unsupported custom key model.")
        hidden, hidden_bias, output = (np.asarray(data[name], dtype=float)
                                       for name in ("hidden", "hidden_bias", "output"))
        size = hidden_bias.size
        output_bias, minor_offset = float(data["output_bias"]), float(data["minor_offset"])
        if (not 1 <= size <= 128 or hidden.shape != (37, size) or hidden_bias.shape != (size,)
                or output.shape != (size,) or not all(np.isfinite(a).all() for a in (hidden, hidden_bias, output))
                or not np.isfinite([output_bias, minor_offset]).all()):
            raise ValueError("Invalid custom key model coefficients.")
        return cls(hidden, hidden_bias, output, output_bias, minor_offset)
