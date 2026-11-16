"""Load the bundled forest and boosting classifiers and score release chroma."""

import hashlib
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np

from .cloud_audio import SAMPLE_RATE
from .keys import KEY_NAMES
from .research_features import release_histograms

ARTIFACT = Path(__file__).with_name("open_tree_models.joblib")
FORMAT = "open-tree-models-v1"
SHA256 = "547a46029c38368425f574e87d23ec324406c18699b4570c8e89464ee0378ca5"


def load_tree_models() -> dict:
    """Verify the tracked artifact before deserializing it."""
    content = ARTIFACT.read_bytes()
    if hashlib.sha256(content).hexdigest() != SHA256:
        raise ValueError("The bundled tree models failed verification.")
    value = joblib.load(BytesIO(content))
    if value["format"] != FORMAT or value["classes"] != KEY_NAMES:
        raise ValueError("Unsupported tree model artifact.")
    return value["models"]


def tree_scores(model, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rank major and minor keys using the same release features as the neural network."""
    features = release_histograms(samples, SAMPLE_RATE)
    if np.ptp(features[0]) < 1e-8:
        raise ValueError("No usable pitch profile was detected.")
    return features[0], model.predict_proba(features.reshape(1, -1))[0]
