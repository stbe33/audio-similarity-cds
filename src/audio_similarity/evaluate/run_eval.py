"""Leave-one-out retrieval evaluation against tag overlap.

For every track we rank all others by cosine similarity and score the ranking
with tag-overlap metrics, per category (genre / instrument / mood/theme) and
overall. A random-ranking floor is computed on the same pool so the numbers are
interpretable: beating the floor is the milestone-1 success bar.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Mapping

import numpy as np

from ..index.search import rank_all
from .metrics import (
    average_jaccard_at_k,
    filter_category,
    jaccard,
    ndcg_at_k,
    precision_at_k,
)

RANDOM_TRIALS = 5  # shuffled rankings averaged for the floor


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def evaluate_index(
    ids: list[str],
    vectors: np.ndarray,
    tracks: Mapping[str, object],
    k: int = 10,
    categories: list[str] | None = None,
    seed: int = 42,
) -> dict:
    """Compute retrieval metrics over all tracks that have metadata."""
    categories = list(categories or ["genre", "instrument", "mood/theme"])
    groups = categories + ["all"]

    # Restrict to tracks we have tags for; keep vectors aligned.
    eval_ids = [i for i in ids if i in tracks]
    pos = {i: p for p, i in enumerate(ids)}
    sub = np.stack([vectors[pos[i]] for i in eval_ids]).astype(np.float32)
    tagsets = {i: set(tracks[i].tags) for i in eval_ids}

    if len(eval_ids) < 2:
        raise ValueError("need at least 2 tracks with metadata to evaluate")

    model_scores = {g: {"precision": [], "jaccard": [], "ndcg": []} for g in groups}
    floor_scores = {g: {"precision": [], "jaccard": []} for g in groups}
    rng = random.Random(seed)

    for qp, qid in enumerate(eval_ids):
        order, _ = rank_all(sub[qp], sub, exclude_index=qp)
        ranked_ids = [eval_ids[j] for j in order]
        q_tags = tagsets[qid]

        for g in groups:
            if g == "all":
                q_c = q_tags
                ranked_c = [tagsets[r] for r in ranked_ids]
            else:
                q_c = filter_category(q_tags, g)
                if not q_c:
                    continue  # query has no tag in this category -> skip
                ranked_c = [filter_category(tagsets[r], g) for r in ranked_ids]

            model_scores[g]["precision"].append(precision_at_k(ranked_c, q_c, k))
            model_scores[g]["jaccard"].append(average_jaccard_at_k(ranked_c, q_c, k))
            rels = [jaccard(ts, q_c) for ts in ranked_c]
            model_scores[g]["ndcg"].append(ndcg_at_k(rels, k, ideal_relevances=rels))

            # Random floor: shuffle the same candidate pool a few times.
            p_trials, j_trials = [], []
            for _ in range(RANDOM_TRIALS):
                shuffled = ranked_c[:]
                rng.shuffle(shuffled)
                p_trials.append(precision_at_k(shuffled, q_c, k))
                j_trials.append(average_jaccard_at_k(shuffled, q_c, k))
            floor_scores[g]["precision"].append(_mean(p_trials))
            floor_scores[g]["jaccard"].append(_mean(j_trials))

    report = {
        "k": k,
        "n_tracks": len(eval_ids),
        "model": {
            g: {
                "precision_at_k": _mean(model_scores[g]["precision"]),
                "avg_jaccard_at_k": _mean(model_scores[g]["jaccard"]),
                "ndcg_at_k": _mean(model_scores[g]["ndcg"]),
                "n_queries": len(model_scores[g]["precision"]),
            }
            for g in groups
        },
        "random_floor": {
            g: {
                "precision_at_k": _mean(floor_scores[g]["precision"]),
                "avg_jaccard_at_k": _mean(floor_scores[g]["jaccard"]),
            }
            for g in groups
        },
    }
    return report


def format_markdown(report: dict) -> str:
    """Render a compact comparison table (model vs random floor)."""
    k = report["k"]
    lines = [
        f"# Retrieval evaluation (k={k}, n={report['n_tracks']})",
        "",
        "| Category | P@k | Floor P@k | Jaccard@k | Floor J@k | nDCG@k | Queries |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for g, m in report["model"].items():
        f = report["random_floor"][g]
        lines.append(
            f"| {g} | {m['precision_at_k']:.3f} | {f['precision_at_k']:.3f} "
            f"| {m['avg_jaccard_at_k']:.3f} | {f['avg_jaccard_at_k']:.3f} "
            f"| {m['ndcg_at_k']:.3f} | {m['n_queries']} |"
        )
    return "\n".join(lines) + "\n"


def write_reports(report: dict, out_dir: str | Path) -> tuple[Path, Path]:
    """Write ``eval_report.json`` and ``eval_report.md``; return both paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "eval_report.json"
    md_path = out_dir / "eval_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(format_markdown(report), encoding="utf-8")
    return json_path, md_path
