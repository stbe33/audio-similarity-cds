"""Select a small, genre-balanced subset of tracks for the baseline.

Full MTG-Jamendo is far too large to embed on a laptop, so we draw ~1-2k tracks.
Sampling is stratified by a track's (first) genre tag and drawn round-robin so no
single genre dominates the retrieval pool or the evaluation.
"""

from __future__ import annotations

import random
from pathlib import Path

from .metadata import TrackMeta


def _stratum(meta: TrackMeta, category: str) -> str:
    cats = sorted(meta.tags_in(category))
    return cats[0] if cats else "__none__"


def stratified_sample(
    tracks: dict[str, TrackMeta],
    n: int,
    category: str = "genre",
    seed: int = 42,
) -> list[str]:
    """Return up to ``n`` track keys, balanced round-robin across ``category``."""
    rng = random.Random(seed)
    groups: dict[str, list[str]] = {}
    for key, meta in tracks.items():
        groups.setdefault(_stratum(meta, category), []).append(key)
    for keys in groups.values():
        rng.shuffle(keys)

    order = sorted(groups.keys())
    selected: list[str] = []
    while len(selected) < n:
        progressed = False
        for g in order:
            if groups[g]:
                selected.append(groups[g].pop())
                progressed = True
                if len(selected) >= n:
                    break
        if not progressed:  # every group exhausted
            break
    return selected


def filter_available(
    keys: list[str],
    tracks: dict[str, TrackMeta],
    audio_root: str | Path,
) -> list[str]:
    """Keep only keys whose audio file exists under ``audio_root``."""
    root = Path(audio_root)
    return [k for k in keys if k in tracks and (root / tracks[k].path).exists()]
