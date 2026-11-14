# SPDX-License-Identifier: AGPL-3.0-only
"""Inference adapters for the two-engine Streamlit build."""

from __future__ import annotations

import hashlib
import re
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier
from threadpoolctl import threadpool_limits

from .baseline import ks_scores
from .cloud_audio import SAMPLE_RATE
from .features import feature_names, histogram_from_audio
from .keys import KEY_NAMES, parse_key

HISTOGRAM = feature_names()[24:]
REPO = "https://github.com/AreteKaiAndreia/Key_Signature"
MAX_MODEL_BYTES = 256 * 1024**2


def rank_three(scores: np.ndarray) -> list[tuple[str, float]]:
    """Return three distinct major/minor labels in deterministic score order."""
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (24,) or not np.isfinite(scores).all():
        raise ValueError("invalid score vector")
    order = np.argsort(-scores, kind="stable")[:3]
    return [(KEY_NAMES[k], float(scores[k])) for k in order]


def essentia_profile(samples: np.ndarray) -> np.ndarray:
    """Aggregate native Essentia HPCP frames from A-first into C-first order."""
    import essentia.standard as es

    frame_size = 4096
    window = es.Windowing(type="hann", size=frame_size)
    spectrum = es.Spectrum(size=frame_size)
    peaks = es.SpectralPeaks(sampleRate=SAMPLE_RATE, minFrequency=40, maxFrequency=5000,
                             magnitudeThreshold=1e-5, maxPeaks=60, orderBy="magnitude")
    hpcp = es.HPCP(size=12, sampleRate=SAMPLE_RATE, referenceFrequency=440,
                   minFrequency=40, maxFrequency=5000, normalized="unitMax",
                   harmonics=0, weightType="cosine", maxShifted=False)
    total = np.zeros(12, dtype=np.float64)
    for frame in es.FrameGenerator(samples, frameSize=frame_size, hopSize=2048, startFromZero=True):
        frequencies, magnitudes = peaks(spectrum(window(frame)))
        if len(frequencies):
            total += hpcp(frequencies, magnitudes)
    mass = total.sum()
    if mass <= 0 or np.ptp(total) <= 1e-8:
        raise ValueError("No usable pitch profile was detected.")
    return np.roll(total / mass, -3)


def essentia_scores(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rank all 24 Krumhansl correlations using Essentia's pitch profile."""
    profile = essentia_profile(samples)
    return profile, ks_scores(profile)


def native_key(profile: np.ndarray) -> tuple[int, float]:
    """Get the reference best key and strength from the native Key API."""
    import essentia.standard as es

    key, scale, strength, _ = es.Key(profileType="krumhansl", usePolyphony=False, pcpSize=12)(
        np.roll(profile, 3).astype(np.float32))
    return parse_key(f"{key} {scale}"), float(strength)


def custom_scores(model, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Score a clip using only the 12-bin chroma distribution."""
    profile = histogram_from_audio(samples, SAMPLE_RATE)
    if np.ptp(profile) < 1e-8:
        raise ValueError("No usable pitch profile was detected.")
    with threadpool_limits(limits=1):
        probabilities = np.asarray(model.predict_proba(profile.reshape(1, -1))[0], dtype=float)
    return profile, probabilities


def _validate_artifact(artifact: dict):
    if (not isinstance(artifact, dict) or artifact.get("feature_names") != HISTOGRAM
            or tuple(artifact.get("classes", ())) != KEY_NAMES):
        raise ValueError("the artifact is not the supported chroma-only model")
    model = artifact.get("model")
    if (not isinstance(model, RandomForestClassifier) or model.n_features_in_ != 12
            or not np.array_equal(model.classes_, np.arange(24))):
        raise ValueError("invalid forest inputs or key classes")
    model.n_jobs = 1
    return model, str(artifact.get("variant", "chroma")), str(artifact.get("version", "local"))


def local_model(path: Path):
    """Load only the project-produced local artifact, never visitor uploads."""
    if not path.is_file():
        raise FileNotFoundError("Train the local deployment artifact first.")
    return _validate_artifact(joblib.load(path))


def released_model(url: str, sha256: str):
    """Verify the authorized, owner-pinned GitHub release before unpickling."""
    if not re.fullmatch(re.escape(REPO) + r"/releases/download/[\w.-]+/[\w.-]+\.joblib", url):
        raise ValueError("model URL must be a pinned release from the project repository")
    if not re.fullmatch(r"[a-f0-9]{64}", sha256):
        raise ValueError("a pinned SHA-256 is required")
    digest = hashlib.sha256()
    with tempfile.NamedTemporaryFile(suffix=".joblib", prefix="keyclf-model-") as file:
        with requests.get(url, stream=True, timeout=(8, 25)) as response:
            response.raise_for_status()
            host = urlparse(response.url).hostname
            if not host or not (host == "github.com" or host.endswith(".githubusercontent.com")):
                raise ValueError("unexpected model download origin")
            size = 0
            for chunk in response.iter_content(1024**2):
                size += len(chunk)
                if size > MAX_MODEL_BYTES:
                    raise ValueError("model exceeds the release size limit")
                digest.update(chunk)
                file.write(chunk)
        if digest.hexdigest() != sha256:
            raise ValueError("model checksum did not match the pinned release")
        file.flush()
        return _validate_artifact(joblib.load(file.name))
