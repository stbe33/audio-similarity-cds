"""Typed configuration loaded from ``config.yaml``.

All stages read their defaults from a single :class:`Config` object so that
experiments (segment length, pooling, model, k, ...) are changed in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AudioConfig:
    target_sr: int = 48000      # overridden by the encoder's required rate
    mono: bool = True


@dataclass
class SegmentConfig:
    window_seconds: float = 10.0        # CLAP is trained on ~10 s clips
    hop_seconds: float = 5.0            # 50% overlap
    min_segment_seconds: float = 3.0    # shorter tails are padded, not dropped


@dataclass
class ModelConfig:
    name: str = "clap"
    checkpoint: str = "laion/larger_clap_music"
    device: str = "auto"                # auto | cpu | mps | cuda
    batch_size: int = 16


@dataclass
class PathsConfig:
    data_dir: str = "data"
    raw_audio_dir: str = "data/raw"
    metadata_dir: str = "data/metadata"
    embeddings_dir: str = "embeddings"
    index_path: str = "embeddings/song_index.npz"
    subset_file: str = "data/subset.txt"


@dataclass
class EvalConfig:
    k: int = 10
    tag_categories: list = field(
        default_factory=lambda: ["genre", "instrument", "mood/theme"]
    )
    seed: int = 42


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # -- construction -------------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Config":
        data = data or {}
        return cls(
            audio=AudioConfig(**data.get("audio", {})),
            segment=SegmentConfig(**data.get("segment", {})),
            model=ModelConfig(**data.get("model", {})),
            paths=PathsConfig(**data.get("paths", {})),
            eval=EvalConfig(**data.get("eval", {})),
        )

    @classmethod
    def load(cls, path: str | Path = "config.yaml") -> "Config":
        """Load config from YAML, falling back to all defaults if absent."""
        p = Path(path)
        if not p.exists():
            return cls()
        with open(p, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data)
