"""Parse MTG-Jamendo autotagging TSV files.

Format (tab-separated, one header row)::

    TRACK_ID  ARTIST_ID  ALBUM_ID  PATH  DURATION  TAGS...

where the trailing columns are tags in ``category---name`` form, e.g.
``genre---rock``, ``instrument---piano``, ``mood/theme---happy``. The on-disk
audio path is ``PATH`` (``XX/TRACKID.mp3``); we key tracks by that file's stem
(the numeric Jamendo id) so the metadata lines up with downloaded audio files.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrackMeta:
    track_id: str          # e.g. "track_1376256"
    path: str              # e.g. "56/1376256.mp3"
    duration: float
    tags: frozenset        # {"genre---rock", "mood/theme---happy", ...}

    @property
    def key(self) -> str:
        """Stable join key = audio file stem (numeric Jamendo id)."""
        return Path(self.path).stem

    def tags_in(self, category: str) -> set:
        prefix = f"{category}---"
        return {t for t in self.tags if t.startswith(prefix)}


def load_autotagging_tsv(tsv_path: str | Path) -> dict[str, TrackMeta]:
    """Load a TSV into ``{key: TrackMeta}`` keyed by audio file stem."""
    tracks: dict[str, TrackMeta] = {}
    with open(tsv_path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        next(reader, None)  # header
        for row in reader:
            if len(row) < 5:
                continue
            path = row[3]
            try:
                duration = float(row[4])
            except (ValueError, IndexError):
                duration = 0.0
            tags = frozenset(t for t in row[5:] if t)
            meta = TrackMeta(track_id=row[0], path=path, duration=duration, tags=tags)
            tracks[meta.key] = meta
    return tracks


def all_categories(tracks: dict[str, TrackMeta]) -> set:
    """Set of tag categories present (the part before ``---``)."""
    cats = set()
    for meta in tracks.values():
        for t in meta.tags:
            if "---" in t:
                cats.add(t.split("---", 1)[0])
    return cats
