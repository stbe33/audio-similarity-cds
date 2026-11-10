"""Command-line entry point: download | subset | embed | query | evaluate | info.

Heavy dependencies (torch/transformers/librosa) are imported only inside the
commands that need them, so ``audio-sim subset`` / ``evaluate`` run without them.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping, Optional

from .config import Config
from .data.metadata import TrackMeta, load_autotagging_tsv

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}


# -- shared helpers ---------------------------------------------------------
def find_metadata_tsv(cfg: Config) -> Optional[Path]:
    meta_dir = Path(cfg.paths.metadata_dir)
    if not meta_dir.exists():
        return None
    preferred = meta_dir / "autotagging.tsv"
    if preferred.exists():
        return preferred
    matches = sorted(meta_dir.glob("autotagging*.tsv"))
    return matches[0] if matches else None


def load_tracks(cfg: Config) -> dict[str, TrackMeta]:
    tsv = find_metadata_tsv(cfg)
    return load_autotagging_tsv(tsv) if tsv else {}


def resolve_corpus(cfg: Config, tracks: Mapping[str, TrackMeta]) -> dict[str, str]:
    """Map track key -> audio path, from the subset file, metadata, or a bare glob."""
    raw = Path(cfg.paths.raw_audio_dir)
    subset_file = Path(cfg.paths.subset_file)

    if tracks and subset_file.exists():
        keys = [ln.strip() for ln in subset_file.read_text().splitlines() if ln.strip()]
        corpus = {}
        for k in keys:
            meta = tracks.get(k)
            if meta and (raw / meta.path).exists():
                corpus[k] = str(raw / meta.path)
        return corpus

    if tracks:
        return {
            k: str(raw / m.path) for k, m in tracks.items() if (raw / m.path).exists()
        }

    # No metadata: treat raw_audio_dir as a plain folder of audio files.
    corpus = {}
    for p in sorted(raw.rglob("*")):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            corpus[p.stem] = str(p)
    return corpus


# -- commands ---------------------------------------------------------------
def cmd_info(cfg: Config, args) -> int:
    tracks = load_tracks(cfg)
    corpus = resolve_corpus(cfg, tracks)
    print(f"model          : {cfg.model.name} ({cfg.model.checkpoint})")
    print(f"segment        : {cfg.segment.window_seconds}s window / "
          f"{cfg.segment.hop_seconds}s hop")
    print(f"metadata tracks: {len(tracks)}")
    print(f"audio available: {len(corpus)}")
    idx = Path(cfg.paths.index_path)
    print(f"index          : {'present' if idx.exists() else 'missing'} ({idx})")
    return 0


def cmd_download_metadata(cfg: Config, args) -> int:
    from .data.download import download_metadata

    out = download_metadata(cfg.paths.metadata_dir, name=args.name)
    tracks = load_autotagging_tsv(out)
    print(f"Saved {out} ({len(tracks)} tracks).")
    return 0


def cmd_subset(cfg: Config, args) -> int:
    from .data.subset import filter_available, stratified_sample

    tracks = load_tracks(cfg)
    if not tracks:
        print("No metadata found. Run: audio-sim download-metadata")
        return 1

    available = filter_available(list(tracks), tracks, cfg.paths.raw_audio_dir)
    pool = {k: tracks[k] for k in (available or list(tracks))}
    if available:
        print(f"{len(available)} tracks have audio; sampling among those.")
    else:
        print("No audio found yet; sampling keys from metadata "
              "(download/copy these tracks before embedding).")

    keys = stratified_sample(pool, args.n, category=args.category, seed=cfg.eval.seed)
    out = Path(cfg.paths.subset_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(keys) + "\n", encoding="utf-8")
    print(f"Wrote {len(keys)} track keys to {out}.")
    return 0


def cmd_embed(cfg: Config, args) -> int:
    from .embed import build_embedder
    from .embed.runner import embed_corpus
    from .index.store import save_index

    tracks = load_tracks(cfg)
    corpus = resolve_corpus(cfg, tracks)
    if not corpus:
        print(f"No audio to embed under {cfg.paths.raw_audio_dir}. "
              "Download audio or run 'subset' first.")
        return 1
    if args.limit:
        corpus = dict(list(corpus.items())[: args.limit])

    print(f"Embedding {len(corpus)} tracks with {cfg.model.name}...")
    embedder = build_embedder(cfg.model)
    ids, matrix = embed_corpus(embedder, corpus, cfg, overwrite=args.overwrite)
    save_index(cfg.paths.index_path, ids, matrix)
    print(f"Indexed {len(ids)} song vectors (dim={matrix.shape[1] if ids else 0}) "
          f"-> {cfg.paths.index_path}")
    return 0


def cmd_query(cfg: Config, args) -> int:
    import numpy as np

    from .aggregate import l2_normalize
    from .index.search import cosine_topk
    from .index.store import load_index

    ids, vectors = load_index(cfg.paths.index_path)
    if not ids:
        print("Empty index. Run 'embed' first.")
        return 1
    tracks = load_tracks(cfg)
    metric = "cos"

    if args.text is not None:
        from .embed import build_embedder

        embedder = build_embedder(cfg.model)
        q = l2_normalize(embedder.embed_text([args.text])[0])
        idx, scores = cosine_topk(q, vectors, k=args.k)
        print(f'Text query: "{args.text}"')
    else:
        if args.key not in ids:
            print(f"Key '{args.key}' not in index. Example keys: {ids[:5]}")
            return 1
        qpos = ids.index(args.key)
        q_tags = sorted(tracks[args.key].tags_in("genre")) if args.key in tracks else []
        print(f"Query: {args.key}   genre: {', '.join(q_tags) or '(n/a)'}")

        if args.rerank == "none":
            idx, scores = cosine_topk(vectors[qpos], vectors, k=args.k, exclude_index=qpos)
        else:
            # Re-rank a larger song-level pool by segment-level similarity so
            # shared-moment matches surface (uses cached per-track segments).
            from .segments import segment_rerank

            pool, _ = cosine_topk(vectors[qpos], vectors,
                                  k=max(args.k, args.pool), exclude_index=qpos)
            pool_keys = [ids[int(j)] for j in pool]
            ranked = segment_rerank(args.key, pool_keys, cfg.paths.embeddings_dir,
                                    cfg.model.name, method=args.rerank)[: args.k]
            idx = np.array([ids.index(k) for k, _ in ranked], dtype=int)
            scores = np.array([s for _, s in ranked], dtype=np.float32)
            metric = f"seg-{args.rerank}"

    print(f"\nTop {len(idx)} neighbors:")
    for rank, (j, s) in enumerate(zip(idx, scores), 1):
        key = ids[int(j)]
        genres = sorted(tracks[key].tags_in("genre")) if key in tracks else []
        tail = f"  [{', '.join(g.split('---')[1] for g in genres)}]" if genres else ""
        print(f"  {rank:2d}. {key}   {metric}={s:.3f}{tail}")
    return 0


def cmd_evaluate(cfg: Config, args) -> int:
    from .evaluate.run_eval import evaluate_index, format_markdown, write_reports
    from .index.store import load_index

    ids, vectors = load_index(cfg.paths.index_path)
    tracks = load_tracks(cfg)
    if not tracks:
        print("Evaluation needs MTG-Jamendo metadata. Run 'download-metadata'.")
        return 1

    report = evaluate_index(
        ids, vectors, tracks, k=args.k,
        categories=cfg.eval.tag_categories, seed=cfg.eval.seed,
    )
    print(format_markdown(report))
    jp, mp = write_reports(report, cfg.paths.embeddings_dir)
    print(f"Reports written: {jp}, {mp}")
    return 0


def cmd_visualize(cfg: Config, args) -> int:
    from . import visualize as viz
    from .index.store import load_index

    ids, vectors = load_index(cfg.paths.index_path)
    if not ids:
        print("Empty index. Run 'embed' first.")
        return 1
    tracks = load_tracks(cfg)
    out_dir = Path(cfg.paths.embeddings_dir) / "figures"
    result = viz.visualize_index(
        ids, vectors, tracks, out_dir,
        k=args.k, method=args.projection, resolution=args.resolution,
    )
    print(f"Found {result['n_communities']} communities "
          f"(modularity={result['modularity']:.3f}); sizes={result['sizes']}\n")
    for name, path in result["paths"].items():
        print(f"  {name:8}: {path}")
    print("\n" + result["markdown"])
    return 0


# -- argument parsing -------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="audio-sim", description=__doc__)
    p.add_argument("--config", default="config.yaml", help="path to config.yaml")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="show config and corpus status").set_defaults(
        func=cmd_info
    )

    dl = sub.add_parser("download-metadata", help="fetch an MTG-Jamendo tag TSV")
    dl.add_argument("--name", default="autotagging",
                    help="autotagging | top50 | genre | instrument | moodtheme")
    dl.set_defaults(func=cmd_download_metadata)

    ss = sub.add_parser("subset", help="select a genre-balanced subset of tracks")
    ss.add_argument("--n", type=int, default=1500, help="number of tracks")
    ss.add_argument("--category", default="genre", help="stratification category")
    ss.set_defaults(func=cmd_subset)

    em = sub.add_parser("embed", help="embed the corpus and build the index")
    em.add_argument("--limit", type=int, default=0, help="cap number of tracks (debug)")
    em.add_argument("--overwrite", action="store_true", help="ignore cached embeddings")
    em.set_defaults(func=cmd_embed)

    q = sub.add_parser("query", help="retrieve neighbors for a track or text prompt")
    g = q.add_mutually_exclusive_group(required=True)
    g.add_argument("--key", help="track key (audio file stem) to query by")
    g.add_argument("--text", help="text prompt, e.g. 'mellow jazz guitar'")
    q.add_argument("--k", type=int, default=10)
    q.add_argument("--rerank", choices=["none", "max", "chamfer", "topk"],
                   default="none",
                   help="re-rank by segment-level similarity (key queries only)")
    q.add_argument("--pool", type=int, default=50,
                   help="song-level candidates to re-rank when --rerank is set")
    q.set_defaults(func=cmd_query)

    ev = sub.add_parser("evaluate", help="tag-overlap evaluation vs a random floor")
    ev.add_argument("--k", type=int, default=10)
    ev.set_defaults(func=cmd_evaluate)

    vz = sub.add_parser("visualize", help="plot song communities from the index")
    vz.add_argument("--k", type=int, default=10, help="neighbors for the kNN graph")
    vz.add_argument("--projection", default="pca", help="pca | tsne | umap")
    vz.add_argument("--resolution", type=float, default=1.0,
                    help="Louvain resolution (higher -> more, smaller communities)")
    vz.set_defaults(func=cmd_visualize)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = Config.load(args.config)
    return args.func(cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
