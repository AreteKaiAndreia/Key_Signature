"""Rank major and minor keys that share a key signature."""

from __future__ import annotations

import numpy as np

from .keys import relative_key


def rank_relative_pairs(scores: np.ndarray) -> list[tuple[int, int, float]]:
    """Rank the 12 relative-key pairs by each pair's highest member score."""
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (24,) or not np.isfinite(scores).all():
        raise ValueError("invalid score vector")
    pairs = []
    seen = set()
    for index in np.argsort(-scores, kind="stable"):
        key = int(index)
        if key in seen:
            continue
        relative = relative_key(key)
        pairs.append((key, relative, float(scores[key])))
        seen.update((key, relative))
    return pairs
