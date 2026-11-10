"""Cosine-similarity retrieval.

At milestone-1 scale (a few thousand songs) a dense normalized matmul is more
than fast enough; swapping in FAISS later only changes this module.
"""

from __future__ import annotations

import numpy as np

from ..aggregate import l2_normalize


def _similarities(
    query: np.ndarray,
    matrix: np.ndarray,
    exclude_index: int | None,
    assume_normalized: bool,
) -> np.ndarray:
    q = np.asarray(query, dtype=np.float32).reshape(-1)
    m = np.asarray(matrix, dtype=np.float32)
    if m.ndim != 2:
        raise ValueError(f"matrix must be 2-D, got shape {m.shape}")
    if not assume_normalized:
        q = l2_normalize(q, axis=-1)
        m = l2_normalize(m, axis=1)
    sims = m @ q
    if exclude_index is not None and 0 <= exclude_index < sims.shape[0]:
        sims = sims.copy()
        sims[exclude_index] = -np.inf
    return sims


def cosine_topk(
    query: np.ndarray,
    matrix: np.ndarray,
    k: int,
    exclude_index: int | None = None,
    assume_normalized: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(indices, scores)`` of the top-``k`` rows by cosine similarity.

    If ``exclude_index`` is given (e.g. the query's own row), it is removed from
    the candidates. Inputs are assumed L2-normalized unless told otherwise.
    """
    sims = _similarities(query, matrix, exclude_index, assume_normalized)
    n = sims.shape[0]
    valid = n - (1 if exclude_index is not None and 0 <= exclude_index < n else 0)
    k = min(k, max(valid, 0))
    if k <= 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=np.float32)

    # Partial sort for the top-k, then order those k by score descending.
    part = np.argpartition(-sims, k - 1)[:k]
    order = part[np.argsort(-sims[part])]
    return order, sims[order].astype(np.float32)


def rank_all(
    query: np.ndarray,
    matrix: np.ndarray,
    exclude_index: int | None = None,
    assume_normalized: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Full descending ranking of every row (used by evaluation)."""
    sims = _similarities(query, matrix, exclude_index, assume_normalized)
    order = np.argsort(-sims)
    if exclude_index is not None and 0 <= exclude_index < sims.shape[0]:
        order = order[order != exclude_index]
    return order, sims[order].astype(np.float32)
