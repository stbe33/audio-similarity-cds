"""Tag-overlap proxy metrics for retrieval quality.

There is no ground-truth "these two songs sound alike" label, so we treat shared
MTG-Jamendo tags as the relevance signal. Tags are strings in ``category---name``
form (e.g. ``genre---rock``), which lets us score each category separately.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

TagSet = Iterable[str]


# -- set-level primitives ---------------------------------------------------
def jaccard(a: TagSet, b: TagSet) -> float:
    """``|a ∩ b| / |a ∪ b|`` (0.0 when both are empty)."""
    a, b = set(a), set(b)
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def shares_tag(a: TagSet, b: TagSet) -> bool:
    """True if the two tag sets share at least one tag."""
    return len(set(a) & set(b)) > 0


def filter_category(tags: TagSet, category: str) -> set:
    """Keep only tags in ``category`` (matched on the ``category---`` prefix)."""
    prefix = f"{category}---"
    return {t for t in tags if t.startswith(prefix)}


# -- ranked-list metrics ----------------------------------------------------
def precision_at_k(
    ranked_tagsets: Sequence[TagSet], query_tags: TagSet, k: int
) -> float:
    """Fraction of the top-``k`` retrieved items sharing >=1 tag with the query."""
    k = min(k, len(ranked_tagsets))
    if k == 0:
        return 0.0
    hits = sum(1 for t in ranked_tagsets[:k] if shares_tag(t, query_tags))
    return hits / k


def average_jaccard_at_k(
    ranked_tagsets: Sequence[TagSet], query_tags: TagSet, k: int
) -> float:
    """Mean Jaccard overlap between the query and the top-``k`` retrieved items."""
    k = min(k, len(ranked_tagsets))
    if k == 0:
        return 0.0
    return float(np.mean([jaccard(t, query_tags) for t in ranked_tagsets[:k]]))


def dcg(relevances: Sequence[float]) -> float:
    """Discounted cumulative gain with log2 position discount."""
    rels = np.asarray(list(relevances), dtype=np.float64)
    if rels.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rels.size + 2))
    return float(np.sum(rels * discounts))


def ndcg_at_k(
    ranked_relevances: Sequence[float],
    k: int,
    ideal_relevances: Sequence[float] | None = None,
) -> float:
    """nDCG@k for graded relevances.

    ``ranked_relevances`` are the graded relevances in retrieved order.
    ``ideal_relevances`` is the pool of achievable relevances used to build the
    ideal ranking; if omitted, the ranked relevances themselves are used.
    """
    ranked = list(ranked_relevances)[:k]
    ideal_pool = list(ideal_relevances) if ideal_relevances is not None else list(
        ranked_relevances
    )
    ideal = sorted(ideal_pool, reverse=True)[:k]
    idcg = dcg(ideal)
    if idcg == 0.0:
        return 0.0
    return dcg(ranked) / idcg
