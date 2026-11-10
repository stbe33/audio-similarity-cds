"""Segment-level comparison between songs.

Song-level mean pooling answers "are these two tracks alike *on average*". It
cannot answer "do these tracks share a *moment*" - a common drop, riff, or
texture - because averaging dilutes any single matching section. These functions
compare the per-segment embeddings directly, which is what the cached
``segments`` arrays (written by ``embed_corpus``) are kept for.

Scorers (all take two ``(n_segments, dim)`` matrices):
- ``max_similarity``   - the single best-matching pair of segments (shared moment)
- ``chamfer_similarity``- each segment's best match, averaged (how much overlaps)
- ``topk_mean_similarity`` - mean of the top-k segment pairs (a few shared moments)

``self_similarity`` gives a song's segment-vs-segment matrix, whose block
structure reveals its sections (verse / chorus / ...).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .aggregate import l2_normalize


def cosine_sim_matrix(
    a: np.ndarray, b: np.ndarray, normalize: bool = False
) -> np.ndarray:
    """Cosine similarity between every segment of ``a`` and every segment of ``b``."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("segment arrays must be 2-D (n_segments, dim)")
    if normalize:
        a = l2_normalize(a, axis=1)
        b = l2_normalize(b, axis=1)
    return a @ b.T


def self_similarity(segments: np.ndarray, normalize: bool = False) -> np.ndarray:
    """A song's ``(n, n)`` segment self-similarity matrix (reveals sections)."""
    return cosine_sim_matrix(segments, segments, normalize)


def max_similarity(a: np.ndarray, b: np.ndarray) -> tuple[float, tuple[int, int]]:
    """Best single segment-pair similarity, with the (i, j) that achieves it."""
    m = cosine_sim_matrix(a, b)
    i, j = np.unravel_index(int(np.argmax(m)), m.shape)
    return float(m[i, j]), (int(i), int(j))


def chamfer_similarity(a: np.ndarray, b: np.ndarray, symmetric: bool = True) -> float:
    """Mean over each segment of its best match in the other song.

    Asymmetric ``a->b`` averages, for every segment of ``a``, its most similar
    segment in ``b``. ``symmetric`` averages both directions.
    """
    m = cosine_sim_matrix(a, b)
    a_to_b = float(m.max(axis=1).mean())
    if not symmetric:
        return a_to_b
    b_to_a = float(m.max(axis=0).mean())
    return 0.5 * (a_to_b + b_to_a)


def topk_mean_similarity(a: np.ndarray, b: np.ndarray, k: int = 3) -> float:
    """Mean of the ``k`` highest segment-pair similarities (a few shared moments)."""
    flat = cosine_sim_matrix(a, b).ravel()
    k = max(1, min(k, flat.size))
    return float(np.partition(flat, -k)[-k:].mean())


# -- working with cached segment embeddings ---------------------------------
def load_segments(
    embeddings_dir: str | Path, model_name: str, key: str
) -> Optional[np.ndarray]:
    """Load the cached ``(n_segments, dim)`` array for one track, or None."""
    path = Path(embeddings_dir) / model_name / f"{key}.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        if "segments" not in data:
            return None
        return np.asarray(data["segments"], dtype=np.float32)


_SCORERS = {
    "max": lambda a, b, k: max_similarity(a, b)[0],
    "chamfer": lambda a, b, k: chamfer_similarity(a, b),
    "topk": lambda a, b, k: topk_mean_similarity(a, b, k),
}


def segment_rerank(
    query_key: str,
    candidate_keys: list[str],
    embeddings_dir: str | Path,
    model_name: str,
    method: str = "max",
    k: int = 3,
) -> list[tuple[str, float]]:
    """Re-score candidates by segment-level similarity to the query.

    Intended to re-rank the top-N song-level neighbors so partial (moment-level)
    matches surface. Candidates without cached segments are skipped.
    """
    if method not in _SCORERS:
        raise ValueError(f"unknown method '{method}' ({sorted(_SCORERS)})")
    scorer = _SCORERS[method]
    qa = load_segments(embeddings_dir, model_name, query_key)
    if qa is None:
        raise ValueError(f"no cached segments for query '{query_key}'")
    qa = l2_normalize(qa, axis=1)

    scored: list[tuple[str, float]] = []
    for key in candidate_keys:
        cb = load_segments(embeddings_dir, model_name, key)
        if cb is None:
            continue
        scored.append((key, scorer(qa, l2_normalize(cb, axis=1), k)))
    scored.sort(key=lambda kv: kv[1], reverse=True)
    return scored
