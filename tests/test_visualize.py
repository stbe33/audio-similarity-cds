import unittest

import numpy as np

from audio_similarity.aggregate import l2_normalize
from audio_similarity.data.metadata import TrackMeta
from audio_similarity.visualize import (
    build_knn_graph,
    community_color,
    detect_communities,
    summarize_communities,
)


def three_blobs(per=15, dim=64, noise=0.03, seed=0):
    rng = np.random.default_rng(seed)
    centers = l2_normalize(rng.standard_normal((3, dim)).astype(np.float32), axis=1)
    ids, vecs, tracks, true = [], [], {}, {}
    i = 0
    for gi, genre in enumerate(["a", "b", "c"]):
        for _ in range(per):
            key = f"s{i}"
            ids.append(key)
            vecs.append(centers[gi] + noise * rng.standard_normal(dim).astype(np.float32))
            tracks[key] = TrackMeta(key, f"00/{i}.mp3", 10.0,
                                    frozenset({f"genre---{genre}"}))
            true[key] = gi
            i += 1
    return ids, l2_normalize(np.stack(vecs), axis=1), tracks, true


class TestVisualize(unittest.TestCase):
    def test_graph_has_edges(self):
        ids, vecs, _, _ = three_blobs()
        g = build_knn_graph(ids, vecs, k=5)
        self.assertEqual(g.number_of_nodes(), len(ids))
        self.assertGreater(g.number_of_edges(), 0)

    def test_communities_recover_blobs(self):
        ids, vecs, _, true = three_blobs()
        g = build_knn_graph(ids, vecs, k=5)
        label_of, communities, modularity = detect_communities(g, seed=1)
        self.assertGreaterEqual(len(communities), 3)
        self.assertGreater(modularity, 0.3)  # clear block structure
        # Each true blob should be (near) homogeneous in assigned community.
        for gi in range(3):
            members = [k for k in ids if true[k] == gi]
            labels = [label_of[k] for k in members]
            dominant = max(set(labels), key=labels.count)
            purity = labels.count(dominant) / len(labels)
            self.assertGreater(purity, 0.9)

    def test_summarize_reports_sizes_and_tags(self):
        ids, vecs, tracks, _ = three_blobs()
        g = build_knn_graph(ids, vecs, k=5)
        _, communities, _ = detect_communities(g, seed=1)
        summary = summarize_communities(communities, tracks)
        self.assertEqual(sum(e["size"] for e in summary), len(ids))
        self.assertTrue(all("genre" in e for e in summary))

    def test_color_folds_overflow_to_gray(self):
        self.assertEqual(community_color(0), "#2a78d6")
        self.assertEqual(community_color(99), "#b4b2ab")


if __name__ == "__main__":
    unittest.main()
