#!/usr/bin/env python3
"""Show what segment-level comparison reveals that song-level pooling hides.

Uses synthetic SEGMENT embeddings with planted section structure (each song is a
sequence of 4 sections x 3 segments; two songs share an identical section). This
isolates the *mechanism* - on real CLAP vectors the same scorers apply, but on
out-of-distribution synthetic *audio* every segment cosine saturates near 1.0,
so this demo controls the embeddings directly instead.

Outputs:
  1. self-similarity of song A (its section structure, as diagonal blocks),
  2. cross-song A-vs-B segment similarity (the shared section lights up), and
  3. a score table where segment-max/chamfer surface a shared section that
     song-level cosine misranks.

    python scripts/segment_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from audio_similarity import segments as seg  # noqa: E402
from audio_similarity import visualize as viz  # noqa: E402
from audio_similarity.aggregate import aggregate_segments, l2_normalize  # noqa: E402

DIM = 128
SEG_PER_SECTION = 3
NOISE = 0.03  # low -> segments within a section match ~0.95 (a clear "moment")
rng = np.random.default_rng(3)


def prototype() -> np.ndarray:
    return l2_normalize(rng.standard_normal(DIM).astype(np.float32))


def section(center: np.ndarray, n: int = SEG_PER_SECTION) -> np.ndarray:
    return np.stack([
        l2_normalize(center + NOISE * rng.standard_normal(DIM).astype(np.float32))
        for _ in range(n)
    ])


def diffuse_section(toward: np.ndarray, alpha: float = 0.5, n: int = SEG_PER_SECTION):
    """Segments that lean partway toward a prototype but never strongly match it."""
    beta = float(np.sqrt(1 - alpha**2))
    return np.stack([
        l2_normalize(alpha * toward + beta * prototype()) for _ in range(n)
    ])


def song(sections) -> np.ndarray:
    return np.concatenate(sections, axis=0)


def main() -> int:
    out_dir = Path(__file__).resolve().parent.parent / "embeddings" / "figures_demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    p = {k: prototype() for k in
         ["intro", "SHARED", "chord", "outro", "b1", "b2", "b3", "c1", "c2", "c3"]}

    A = song([section(p["intro"]), section(p["SHARED"]),
              section(p["chord"]), section(p["outro"])])
    B = song([section(p["b1"]), section(p["b2"]),
              section(p["SHARED"]), section(p["b3"])])           # shares SHARED
    C = song([section(p["c1"]), section(p["chord"]),
              section(p["c2"]), section(p["c3"])])               # shares chord
    # D: every section leans halfway toward one of A's sections -> its *mean*
    # aligns with A, but no single section strongly matches any of A's.
    D = song([diffuse_section(p["intro"]), diffuse_section(p["chord"]),
              diffuse_section(p["outro"]), diffuse_section(prototype())])

    labels = ["intro", "SHARED", "chord", "outro"]
    bnds = [3, 6, 9]
    mids = [1, 4, 7, 10]

    # 1) self-similarity of A -> four diagonal blocks (sections).
    viz.segment_heatmap(
        seg.self_similarity(A), out_dir / "segment_selfsim.png",
        title="Self-similarity of song A (sections)",
        boundaries=bnds, xticks=(mids, labels), yticks=(mids, labels),
    )

    # 2) A vs B segments -> the shared section lights up off the diagonal.
    viz.segment_heatmap(
        seg.cosine_sim_matrix(A, B), out_dir / "segment_cross.png",
        title="A vs B segments (shared section lights up)",
        xlabel="B segment", ylabel="A segment",
        xticks=(mids, ["b1", "b2", "SHARED", "b3"]), yticks=(mids, labels),
    )

    # 3) song-level vs segment-level scoring.
    song_a = aggregate_segments(A, normalize=True)
    print(f"  query = A  (sections: {labels})\n")
    print(f"  {'candidate':22} {'song-cos':>9} {'seg-max':>9} {'chamfer':>9}  best-pair")
    for name, X, note in [
        ("B (shares SHARED)", B, "b"),
        ("C (shares chord)", C, "c"),
        ("D (diffuse, no match)", D, "d"),
    ]:
        song_cos = float(aggregate_segments(X, normalize=True) @ song_a)
        smax, (i, j) = seg.max_similarity(A, X)
        cham = seg.chamfer_similarity(A, X)
        print(f"  {name:22} {song_cos:9.3f} {smax:9.3f} {cham:9.3f}  "
              f"A.seg{i} <-> {note}.seg{j}")

    print(f"\n  figures: {out_dir/'segment_selfsim.png'}, {out_dir/'segment_cross.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
