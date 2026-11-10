"""Fetch MTG-Jamendo metadata, and help acquire a subset of the audio.

Metadata TSVs are small and live in the GitHub repo, so we download them
directly. Audio ships as large TAR shards served by the official
``scripts/download/download.py`` in the MTG-Jamendo repo; we wrap that script
(and provide a subset copier) rather than re-implement shard URLs that may change.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

from .metadata import TrackMeta

RAW_BASE = "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/data/"

# name -> filename in the repo's data/ folder
TSV_FILES = {
    "autotagging": "autotagging.tsv",
    "top50": "autotagging_top50tags.tsv",
    "genre": "autotagging_genre.tsv",
    "instrument": "autotagging_instrument.tsv",
    "moodtheme": "autotagging_moodtheme.tsv",
}


def download_metadata(dest_dir: str | Path, name: str = "autotagging") -> Path:
    """Download one metadata TSV into ``dest_dir`` and return its path."""
    if name not in TSV_FILES:
        raise ValueError(f"unknown metadata '{name}', choose from {sorted(TSV_FILES)}")
    filename = TSV_FILES[name]
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / filename
    url = RAW_BASE + filename
    print(f"Downloading {url} -> {out}")
    with urllib.request.urlopen(url) as resp, open(out, "wb") as fh:  # noqa: S310
        shutil.copyfileobj(resp, fh)
    return out


def run_official_download(
    repo_dir: str | Path,
    output_dir: str | Path,
    dataset: str = "autotagging_moodtheme",
    type_: str = "audio-low",
    source: str = "mtg-fast",
    unpack: bool = True,
    remove: bool = True,
) -> int:
    """Invoke the cloned MTG-Jamendo ``scripts/download/download.py``.

    Clone it first::  git clone https://github.com/MTG/mtg-jamendo-dataset
    ``type_='audio-low'`` (mono VBR mp3) is the smallest full-audio option, and
    ``dataset='autotagging_moodtheme'`` is the smallest labeled subset (~46 GB).
    """
    script = Path(repo_dir) / "scripts" / "download" / "download.py"
    if not script.exists():
        raise FileNotFoundError(
            f"{script} not found. Clone the MTG-Jamendo repo first:\n"
            "  git clone https://github.com/MTG/mtg-jamendo-dataset"
        )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(script),
        "--dataset", dataset,
        "--type", type_,
        "--from", source,
        str(output_dir),
    ]
    if unpack:
        cmd.append("--unpack")
    if remove:
        cmd.append("--remove")
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


def copy_subset(
    keys: list[str],
    tracks: dict[str, TrackMeta],
    source_dir: str | Path,
    dest_dir: str | Path,
) -> int:
    """Copy only the selected tracks from a full download into ``dest_dir``.

    Preserves the ``XX/TRACKID.mp3`` layout so metadata paths keep matching.
    Returns the number of files copied.
    """
    source_dir, dest_dir = Path(source_dir), Path(dest_dir)
    copied = 0
    for key in keys:
        meta = tracks.get(key)
        if meta is None:
            continue
        src = source_dir / meta.path
        if not src.exists():
            continue
        dst = dest_dir / meta.path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    return copied
