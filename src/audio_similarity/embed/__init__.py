"""Pretrained audio encoders behind a common, swappable interface."""

from __future__ import annotations

from ..config import ModelConfig
from .base import Embedder


def build_embedder(cfg: ModelConfig) -> Embedder:
    """Construct the encoder named in config. New encoders register here."""
    name = cfg.name.lower()
    if name == "clap":
        from .clap import ClapEmbedder

        return ClapEmbedder(
            checkpoint=cfg.checkpoint, device=cfg.device, batch_size=cfg.batch_size
        )
    # Discogs-EffNet (TensorFlow/Essentia) is milestone 2.
    raise ValueError(f"unknown model '{cfg.name}' (available: clap)")


__all__ = ["Embedder", "build_embedder"]
