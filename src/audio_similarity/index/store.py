"""Persist and load the song-level embedding index as a single ``.npz`` file."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_index(path: str | Path, ids: list[str], vectors: np.ndarray) -> Path:
    """Save track ids and their aligned embedding matrix to ``path``."""
    ids = [str(i) for i in ids]
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"vectors must be 2-D, got shape {vectors.shape}")
    if len(ids) != vectors.shape[0]:
        raise ValueError(
            f"ids ({len(ids)}) and vectors ({vectors.shape[0]}) length mismatch"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ids=np.array(ids), vectors=vectors)
    return path


def load_index(path: str | Path) -> tuple[list[str], np.ndarray]:
    """Load ``(ids, vectors)`` previously written by :func:`save_index`."""
    data = np.load(Path(path), allow_pickle=False)
    ids = [str(x) for x in data["ids"].tolist()]
    vectors = np.asarray(data["vectors"], dtype=np.float32)
    return ids, vectors
