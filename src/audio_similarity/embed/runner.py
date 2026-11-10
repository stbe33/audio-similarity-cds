"""Drive encoders over a corpus: load -> segment -> embed -> cache -> index.

Per-track embeddings are cached as ``<embeddings>/<model>/<key>.npz`` holding both
the segment matrix (kept for the future segment+song fusion experiment) and the
pooled song vector, so re-runs skip already-embedded tracks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import numpy as np

from ..aggregate import aggregate_segments
from ..config import Config
from ..preprocess.loader import load_audio
from ..preprocess.segment import frame_signal
from .base import Embedder


def embed_track(
    embedder: Embedder, audio_path: str | Path, cfg: Config
) -> Optional[np.ndarray]:
    """Return the ``(n_segments, dim)`` embeddings for one track, or None if empty."""
    y = load_audio(audio_path, embedder.target_sr, cfg.audio.mono)
    segments = frame_signal(
        y,
        embedder.target_sr,
        cfg.segment.window_seconds,
        cfg.segment.hop_seconds,
        cfg.segment.min_segment_seconds,
    )
    if segments.shape[0] == 0:
        return None
    return embedder.embed_segments(segments)


def embed_corpus(
    embedder: Embedder,
    key_to_path: Mapping[str, str],
    cfg: Config,
    overwrite: bool = False,
    progress: bool = True,
) -> tuple[list[str], np.ndarray]:
    """Embed every track, caching per track, and return ``(ids, song_matrix)``."""
    cache_dir = Path(cfg.paths.embeddings_dir) / embedder.name
    cache_dir.mkdir(parents=True, exist_ok=True)

    items = list(key_to_path.items())
    iterator = items
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(items, desc=f"embed[{embedder.name}]")
        except ImportError:
            pass

    ids: list[str] = []
    song_vecs: list[np.ndarray] = []
    for key, path in iterator:
        cache = cache_dir / f"{key}.npz"
        if cache.exists() and not overwrite:
            song = np.load(cache)["song"]
        else:
            seg_vecs = embed_track(embedder, path, cfg)
            if seg_vecs is None:
                continue
            song = aggregate_segments(seg_vecs, normalize=True)
            np.savez(
                cache,
                segments=seg_vecs.astype(np.float32),
                song=song.astype(np.float32),
            )
        ids.append(key)
        song_vecs.append(song)

    if song_vecs:
        matrix = np.stack(song_vecs).astype(np.float32)
    else:
        matrix = np.zeros((0, getattr(embedder, "dim", 0) or 1), dtype=np.float32)
    return ids, matrix
