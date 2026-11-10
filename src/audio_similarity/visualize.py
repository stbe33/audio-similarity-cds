"""Visualize communities of similar songs from the embedding index.

Three complementary views, all driven by cosine similarity of the song vectors:

1. ``community_graph``   - a mutual-kNN similarity graph, nodes colored by the
   community a modularity (Louvain) partition assigns them.
2. ``projection_scatter``- a 2-D projection (PCA / t-SNE / UMAP) of the vectors,
   colored by community (or by genre, if metadata is supplied).
3. ``similarity_heatmap``- the cosine matrix reordered by community, whose block
   structure *is* the communities.

Plus ``summarize_communities`` for the dominant tags per community.

Colors come from a CVD-validated categorical palette; identity is carried by a
legend and centroid labels, never color alone. matplotlib/networkx/sklearn are
imported lazily so importing this module stays cheap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

# -- validated palette (see dataviz skill references/palette.md) -------------
COMMUNITY_COLORS = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#6250d6", "#e34948",
]
OTHER_COLOR = "#b4b2ab"          # 9th+ community folds to a muted gray
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
SEQUENTIAL = "Blues"             # single-hue magnitude ramp for the heatmap


def community_color(label: int) -> str:
    return COMMUNITY_COLORS[label] if 0 <= label < len(COMMUNITY_COLORS) else OTHER_COLOR


# -- similarity graph + community detection ---------------------------------
def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    """Full cosine-similarity matrix (vectors assumed L2-normalized)."""
    v = np.asarray(vectors, dtype=np.float32)
    return v @ v.T


def build_knn_graph(
    ids: Sequence[str],
    vectors: np.ndarray,
    k: int = 10,
    mutual: bool = True,
    min_sim: float = 0.0,
):
    """Build a (mutual) k-NN cosine graph; edge weight is the cosine similarity."""
    import networkx as nx

    ids = list(ids)
    n = len(ids)
    sim = cosine_matrix(vectors)
    np.fill_diagonal(sim, -np.inf)
    k = max(1, min(k, n - 1))
    nn = np.argsort(-sim, axis=1)[:, :k]
    neighbor_of = [set(nn[i]) for i in range(n)]

    graph = nx.Graph()
    graph.add_nodes_from(ids)
    for i in range(n):
        for j in nn[i]:
            s = float(sim[i, j])
            if s < min_sim:
                continue
            if mutual and i not in neighbor_of[j]:
                continue
            a, b = (i, j) if i < j else (j, i)
            w = graph.get_edge_data(ids[a], ids[b], {}).get("weight", -1.0)
            if s > w:
                graph.add_edge(ids[a], ids[b], weight=s)

    # Rescue isolated nodes (common with mutual-kNN on fringe points): link each
    # to its single nearest neighbor so it folds into that community instead of
    # becoming a one-song singleton.
    for i in range(n):
        if graph.degree(ids[i]) == 0:
            j = int(nn[i][0])
            graph.add_edge(ids[i], ids[j], weight=float(sim[i, j]))
    return graph


def detect_communities(graph, resolution: float = 1.0, seed: int = 42):
    """Partition the graph into communities (Louvain, greedy-modularity fallback).

    Returns ``(label_of, communities, modularity)`` where ``label_of`` maps each
    node id to an integer community label (0 = largest community).
    """
    import networkx as nx

    try:
        communities = nx.community.louvain_communities(
            graph, weight="weight", resolution=resolution, seed=seed
        )
    except Exception:
        communities = nx.community.greedy_modularity_communities(graph, weight="weight")

    communities = sorted((set(c) for c in communities), key=len, reverse=True)
    label_of: dict[str, int] = {}
    for label, nodes in enumerate(communities):
        for node in nodes:
            label_of[node] = label
    try:
        modularity = nx.community.modularity(graph, communities, weight="weight")
    except Exception:
        modularity = float("nan")
    return label_of, communities, modularity


# -- 2-D projection ----------------------------------------------------------
def project_2d(vectors: np.ndarray, method: str = "pca", seed: int = 42) -> np.ndarray:
    """Project embeddings to 2-D with PCA (default), t-SNE, or UMAP."""
    v = np.asarray(vectors, dtype=np.float32)
    method = method.lower()
    if method == "pca":
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=seed).fit_transform(v)
    if method == "tsne":
        from sklearn.manifold import TSNE

        perplexity = float(min(30, max(5, (len(v) - 1) / 3)))
        return TSNE(
            n_components=2, random_state=seed, perplexity=perplexity, init="pca"
        ).fit_transform(v)
    if method == "umap":
        import umap  # optional dependency

        return umap.UMAP(n_components=2, random_state=seed).fit_transform(v)
    raise ValueError(f"unknown projection '{method}' (pca | tsne | umap)")


# -- tag summary -------------------------------------------------------------
def summarize_communities(
    communities: Sequence[set],
    tracks: Optional[Mapping[str, object]] = None,
    categories: Sequence[str] = ("genre", "instrument", "mood/theme"),
    top_n: int = 5,
) -> list[dict]:
    """Per community: size and the most common tags in each category."""
    from collections import Counter

    out = []
    for label, nodes in enumerate(communities):
        entry: dict = {"community": label, "size": len(nodes)}
        if tracks:
            for cat in categories:
                counter: Counter = Counter()
                for node in nodes:
                    meta = tracks.get(node)
                    if meta is None:
                        continue
                    for tag in meta.tags_in(cat):
                        counter[tag.split("---", 1)[1]] += 1
                entry[cat] = counter.most_common(top_n)
        out.append(entry)
    return out


def summary_markdown(summary: Sequence[dict], modularity: float) -> str:
    lines = [
        f"# Song communities ({len(summary)} found, modularity={modularity:.3f})",
        "",
        "| Community | Size | Top genres | Top moods |",
        "| --- | ---: | --- | --- |",
    ]
    for e in summary:
        g = ", ".join(f"{t} ({c})" for t, c in e.get("genre", [])[:3]) or "-"
        m = ", ".join(f"{t} ({c})" for t, c in e.get("mood/theme", [])[:3]) or "-"
        lines.append(f"| C{e['community']} | {e['size']} | {g} | {m} |")
    return "\n".join(lines) + "\n"


# -- plotting ----------------------------------------------------------------
def _pyplot():
    import matplotlib

    matplotlib.use("Agg")  # headless, file output only
    import matplotlib.pyplot as plt

    return plt


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)


def _legend_handles(plt, labels_present: Sequence[int], summary=None):
    from matplotlib.lines import Line2D

    handles = []
    for label in labels_present:
        text = f"C{label}"
        if summary is not None and label < len(summary):
            genres = summary[label].get("genre", [])
            if genres:
                text = f"C{label}: {genres[0][0]}"
        handles.append(
            Line2D([0], [0], marker="o", linestyle="", markersize=8,
                   markerfacecolor=community_color(label),
                   markeredgecolor="white", markeredgewidth=0.6, label=text)
        )
    return handles


def projection_scatter(
    coords: np.ndarray,
    label_of: Mapping[str, int],
    ids: Sequence[str],
    out_path: str | Path,
    title: str = "Song communities (2-D projection)",
    summary=None,
) -> Path:
    plt = _pyplot()
    labels = np.array([label_of.get(i, -1) for i in ids])
    present = sorted(set(labels.tolist()))
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=SURFACE)
    y_margin = 0.04 * (coords[:, 1].max() - coords[:, 1].min() + 1e-9)
    for label in present:
        m = labels == label
        ax.scatter(coords[m, 0], coords[m, 1], s=42, alpha=0.85, zorder=2,
                   color=community_color(label), edgecolors="white", linewidths=0.5)
        # Centroid label just above the cluster -> identity never rests on color
        # alone, and the box does not sit on top of the points.
        cx = coords[m, 0].mean()
        cy = coords[m, 1].max() + y_margin
        ax.text(cx, cy, f"C{label}", fontsize=9, fontweight="bold", color=INK,
                ha="center", va="bottom", zorder=3,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec=AXIS, alpha=0.65))
    _style_axes(ax)
    ax.set_title(title, color=INK, fontsize=13, fontweight="bold", loc="left")
    ax.set_xlabel("dim 1", color=SECONDARY, fontsize=9)
    ax.set_ylabel("dim 2", color=SECONDARY, fontsize=9)
    ax.legend(handles=_legend_handles(plt, present, summary), loc="center left",
              bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8,
              title="community", title_fontsize=9)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out_path


def community_graph(
    graph,
    label_of: Mapping[str, int],
    out_path: str | Path,
    title: str = "Song similarity graph (communities)",
    pos: Optional[dict] = None,
    seed: int = 42,
    summary=None,
) -> Path:
    import networkx as nx

    plt = _pyplot()
    if pos is None:
        pos = nx.spring_layout(graph, weight="weight", seed=seed)
    fig, ax = plt.subplots(figsize=(8, 7), facecolor=SURFACE)
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=GRID, width=0.6, alpha=0.7)
    node_colors = [community_color(label_of.get(n, -1)) for n in graph.nodes()]
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_colors, node_size=70,
                           edgecolors="white", linewidths=0.5)
    present = sorted(set(label_of.values()))
    ax.legend(handles=_legend_handles(plt, present, summary), loc="center left",
              bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8,
              title="community", title_fontsize=9)
    ax.set_title(title, color=INK, fontsize=13, fontweight="bold", loc="left")
    ax.set_axis_off()
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out_path


def similarity_heatmap(
    ids: Sequence[str],
    vectors: np.ndarray,
    label_of: Mapping[str, int],
    out_path: str | Path,
    title: str = "Cosine similarity, ordered by community",
) -> Path:
    plt = _pyplot()
    ids = list(ids)
    order = sorted(range(len(ids)), key=lambda i: (label_of.get(ids[i], -1), i))
    sim = cosine_matrix(vectors)[np.ix_(order, order)]
    offdiag = sim[~np.eye(len(ids), dtype=bool)]
    vmin = float(np.percentile(offdiag, 2)) if offdiag.size else 0.0

    fig, ax = plt.subplots(figsize=(7, 6), facecolor=SURFACE)
    im = ax.imshow(sim, cmap=SEQUENTIAL, vmin=vmin, vmax=1.0, interpolation="nearest")
    # Community boundary lines.
    labels_sorted = [label_of.get(ids[i], -1) for i in order]
    boundaries = [i for i in range(1, len(labels_sorted))
                  if labels_sorted[i] != labels_sorted[i - 1]]
    for b in boundaries:
        ax.axhline(b - 0.5, color=INK, linewidth=0.5, alpha=0.5)
        ax.axvline(b - 0.5, color=INK, linewidth=0.5, alpha=0.5)
    ax.set_title(title, color=INK, fontsize=13, fontweight="bold", loc="left")
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("cosine similarity", color=SECONDARY, fontsize=9)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out_path


# -- orchestrator ------------------------------------------------------------
def visualize_index(
    ids: Sequence[str],
    vectors: np.ndarray,
    tracks: Optional[Mapping[str, object]],
    out_dir: str | Path,
    k: int = 10,
    method: str = "pca",
    resolution: float = 1.0,
    seed: int = 42,
) -> dict:
    """Detect communities and write all three figures + a tag summary."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    graph = build_knn_graph(ids, vectors, k=k, mutual=True)
    label_of, communities, modularity = detect_communities(
        graph, resolution=resolution, seed=seed
    )
    summary = summarize_communities(communities, tracks)
    coords = project_2d(vectors, method=method, seed=seed)

    paths = {
        "scatter": projection_scatter(
            coords, label_of, ids, out_dir / f"communities_{method}.png", summary=summary
        ),
        "graph": community_graph(graph, label_of, out_dir / "communities_graph.png",
                                 seed=seed, summary=summary),
        "heatmap": similarity_heatmap(ids, vectors, label_of,
                                      out_dir / "communities_heatmap.png"),
    }
    md = summary_markdown(summary, modularity)
    md_path = out_dir / "communities.md"
    md_path.write_text(md, encoding="utf-8")
    paths["summary"] = md_path

    return {
        "n_communities": len(communities),
        "modularity": modularity,
        "sizes": [len(c) for c in communities],
        "paths": {kk: str(v) for kk, v in paths.items()},
        "summary": summary,
        "markdown": md,
    }
