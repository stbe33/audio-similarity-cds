#!/usr/bin/env python3
"""Smoke test: load CLAP and inspect what the embedding vectors look like.

Synthesizes a few distinct 10 s / 48 kHz clips directly as numpy arrays (no
audio files needed) and runs them through the real ClapEmbedder, printing the
vector shape, scale, and the cosine-similarity structure that drives retrieval.

    python scripts/clap_smoke.py            # from the repo root
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from audio_similarity.aggregate import aggregate_segments, l2_normalize  # noqa: E402
from audio_similarity.config import Config  # noqa: E402
from audio_similarity.embed.clap import ClapEmbedder  # noqa: E402

SR, DUR = 48000, 10.0
_t = np.arange(int(SR * DUR), dtype=np.float32) / SR
_rng = np.random.default_rng(0)


def sine(f: float) -> np.ndarray:
    return (0.5 * np.sin(2 * np.pi * f * _t)).astype(np.float32)


def chirp(f0: float, f1: float) -> np.ndarray:
    k = (f1 - f0) / DUR
    return (0.5 * np.sin(2 * np.pi * (f0 * _t + 0.5 * k * _t * _t))).astype(np.float32)


def main() -> int:
    cfg = Config.load()
    clips = {
        "sine_440Hz": sine(440),
        "sine_442Hz": sine(442),           # ~identical to 440
        "sine_220Hz": sine(220),           # octave below
        "white_noise": (0.3 * _rng.standard_normal(_t.size)).astype(np.float32),
        "chirp_200_2000": chirp(200, 2000),
    }
    names = list(clips)
    batch = np.stack([clips[n] for n in names])

    print(f"Loading {cfg.model.checkpoint} (first run downloads/converts weights)...")
    emb = ClapEmbedder(checkpoint=cfg.model.checkpoint, device=cfg.model.device)
    print(f"device={emb.device}  dim={emb.dim}  target_sr={emb.target_sr}")

    vecs = emb.embed_segments(batch)  # (5, dim)
    np.set_printoptions(precision=4, suppress=True, linewidth=120)
    print("\nsegment embeddings :", vecs.shape, vecs.dtype)
    print("per-clip L2 norm   :", np.linalg.norm(vecs, axis=1))
    print("value stats        : min %.3f max %.3f mean %.4f std %.4f"
          % (vecs.min(), vecs.max(), vecs.mean(), vecs.std()))

    song0 = aggregate_segments(vecs[:1], normalize=True)
    print("song vector norm   :", round(float(np.linalg.norm(song0)), 6))

    vn = l2_normalize(vecs, axis=1)
    sim = vn @ vn.T
    print("\ncosine similarity (higher = more alike):")
    print("            " + "".join(f"{n[:10]:>12}" for n in names))
    for i, n in enumerate(names):
        print(f"{n[:10]:>11} " + "".join(f"{sim[i, j]:>12.3f}" for j in range(len(names))))

    print("\nnote: CLAP cosines sit in a narrow high band (anisotropy) — rank order "
          "is the signal, not the absolute value.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
