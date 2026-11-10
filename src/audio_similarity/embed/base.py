"""The ``Embedder`` interface every encoder implements.

Keeping the contract tiny (segments in, vectors out, at a declared sample rate)
is what lets CLAP, Discogs-EffNet, and future models be swapped without touching
the rest of the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Embedder(ABC):
    name: str = "base"
    target_sr: int = 48000   # the sample rate segments must be provided at
    dim: int = 0             # embedding dimensionality

    @abstractmethod
    def embed_segments(self, segments: np.ndarray) -> np.ndarray:
        """Map a ``(n_segments, window_samples)`` batch to ``(n_segments, dim)``."""
        raise NotImplementedError

    def embed_text(self, texts: list[str]) -> np.ndarray:
        """Optional: map text prompts into the same space (CLAP supports this)."""
        raise NotImplementedError(f"{self.name} does not support text queries")


def resolve_device(preference: str = "auto") -> str:
    """Pick a torch device: explicit preference, else cuda > mps > cpu."""
    import torch

    if preference and preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def iter_batches(array: np.ndarray, batch_size: int):
    """Yield consecutive row-batches of ``array``."""
    for start in range(0, len(array), batch_size):
        yield array[start : start + batch_size]
