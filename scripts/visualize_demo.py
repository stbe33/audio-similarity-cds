#!/usr/bin/env python3
"""Generate example community visualizations on SYNTHETIC embeddings.

Real song embeddings need audio downloaded + `audio-sim embed`; this script
fabricates genre-structured vectors (each genre a direction in 256-d, plus
noise) with synthetic tags so you can see exactly what `audio-sim visualize`
produces. Figures are written to embeddings/figures_demo/.

    python scripts/visualize_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from audio_similarity import visualize as viz  # noqa: E402
from audio_similarity.aggregate import l2_normalize  # noqa: E402
from audio_similarity.data.metadata import TrackMeta  # noqa: E402

DIM = 256
PER_GENRE = 30
NOISE = 0.09  # loose enough that each genre is a visible cloud, still separable
GENRES = ["ambient", "techno", "folk", "metal", "jazz", "hiphop"]
MOODS = {
    "ambient": "calm", "techno": "energetic", "folk": "warm",
    "metal": "dark", "jazz": "smooth", "hiphop": "groovy",
}
INSTR = {
    "ambient": "synthesizer", "techno": "synthesizer", "folk": "guitar",
    "metal": "guitar", "jazz": "saxophone", "hiphop": "drum machine",
}


def main() -> int:
    rng = np.random.default_rng(7)
    centers = l2_normalize(
        rng.standard_normal((len(GENRES), DIM)).astype(np.float32), axis=1
    )
    ids, vecs, tracks = [], [], {}
    i = 0
    for gi, genre in enumerate(GENRES):
        for _ in range(PER_GENRE):
            v = centers[gi] + NOISE * rng.standard_normal(DIM).astype(np.float32)
            key = f"song{i:03d}"
            tags = {f"genre---{genre}", f"mood/theme---{MOODS[genre]}"}
            if rng.random() < 0.6:
                tags.add(f"instrument---{INSTR[genre]}")
            tracks[key] = TrackMeta(key, f"00/{i}.mp3", 10.0, frozenset(tags))
            ids.append(key)
            vecs.append(v)
            i += 1
    vectors = l2_normalize(np.stack(vecs), axis=1)

    out_dir = Path(__file__).resolve().parent.parent / "embeddings" / "figures_demo"
    print(f"Visualizing {len(ids)} synthetic songs across {len(GENRES)} genres...")
    result = viz.visualize_index(
        ids, vectors, tracks, out_dir, k=8, method="tsne", resolution=1.0
    )
    print(f"\nFound {result['n_communities']} communities "
          f"(modularity={result['modularity']:.3f}); sizes={result['sizes']}")
    for name, path in result["paths"].items():
        print(f"  {name:8}: {path}")
    print("\n" + result["markdown"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
