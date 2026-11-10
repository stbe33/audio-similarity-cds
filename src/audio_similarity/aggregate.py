"""Aggregate per-segment embeddings into a single song-level vector.

Milestone 1 uses mean pooling followed by L2 normalization, so that cosine
similarity reduces to a dot product downstream. Other poolings (max, attention)
are deferred to the experiments milestone.
"""

from __future__ import annotations

import numpy as np


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """Scale vectors to unit L2 norm along ``axis`` (zero vectors are left near zero)."""
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return (x / np.maximum(norm, eps)).astype(np.float32)


def mean_pool(segment_vectors: np.ndarray) -> np.ndarray:
    """Mean of segment embeddings: ``(n_segments, dim) -> (dim,)``."""
    sv = np.asarray(segment_vectors, dtype=np.float32)
    if sv.ndim != 2:
        raise ValueError(f"expected 2-D (n_segments, dim), got shape {sv.shape}")
    if sv.shape[0] == 0:
        raise ValueError("cannot pool zero segments")
    return sv.mean(axis=0).astype(np.float32)


def aggregate_segments(
    segment_vectors: np.ndarray, normalize: bool = True
) -> np.ndarray:
    """Mean-pool segment vectors into a song vector, L2-normalized by default."""
    song = mean_pool(segment_vectors)
    if normalize:
        song = l2_normalize(song, axis=-1)
    return song.astype(np.float32)
