import tempfile
import unittest
from pathlib import Path

import numpy as np

from audio_similarity.aggregate import l2_normalize
from audio_similarity.segments import (
    chamfer_similarity,
    cosine_sim_matrix,
    load_segments,
    max_similarity,
    segment_rerank,
    self_similarity,
    topk_mean_similarity,
)


def unit(rows):
    return l2_normalize(np.asarray(rows, dtype=np.float32), axis=1)


class TestSegmentScorers(unittest.TestCase):
    def setUp(self):
        # a: two orthogonal directions; b shares a's second direction exactly.
        self.a = unit([[1, 0, 0], [0, 1, 0]])
        self.b = unit([[0, 0, 1], [0, 1, 0]])  # b[1] == a[1]

    def test_sim_matrix_shape(self):
        m = cosine_sim_matrix(self.a, self.b)
        self.assertEqual(m.shape, (2, 2))

    def test_self_similarity_diagonal_is_one(self):
        s = self_similarity(self.a)
        np.testing.assert_allclose(np.diag(s), [1.0, 1.0], atol=1e-6)

    def test_max_similarity_finds_shared_segment(self):
        score, (i, j) = max_similarity(self.a, self.b)
        self.assertAlmostEqual(score, 1.0, places=5)
        self.assertEqual((i, j), (1, 1))  # the identical pair

    def test_chamfer_symmetric(self):
        self.assertAlmostEqual(
            chamfer_similarity(self.a, self.b), chamfer_similarity(self.b, self.a),
            places=6,
        )

    def test_topk_mean_between_min_and_max(self):
        m = cosine_sim_matrix(self.a, self.b)
        tk = topk_mean_similarity(self.a, self.b, k=2)
        self.assertLessEqual(tk, m.max() + 1e-6)
        self.assertGreaterEqual(tk, m.min() - 1e-6)

    def test_max_beats_songlevel_for_partial_match(self):
        # Song mean pooling dilutes the shared direction; segment-max keeps it.
        song_a = l2_normalize(self.a.mean(0))
        song_b = l2_normalize(self.b.mean(0))
        song_cos = float(song_a @ song_b)
        seg_max, _ = max_similarity(self.a, self.b)
        self.assertGreater(seg_max, song_cos)


class TestSegmentCache(unittest.TestCase):
    def test_load_and_rerank(self):
        with tempfile.TemporaryDirectory() as d:
            model_dir = Path(d) / "clap"
            model_dir.mkdir()
            a = unit([[1, 0], [0, 1]])
            shares = unit([[0.1, 1.0], [1.0, 0.2]])   # strong match to a[1]
            weak = unit([[1.0, 0.9], [0.9, 1.0]])     # diffuse
            for key, segs in {"q": a, "shares": shares, "weak": weak}.items():
                np.savez(model_dir / f"{key}.npz", segments=segs)

            self.assertEqual(load_segments(d, "clap", "q").shape, (2, 2))
            self.assertIsNone(load_segments(d, "clap", "missing"))

            ranked = segment_rerank("q", ["weak", "shares"], d, "clap", method="max")
            self.assertEqual(ranked[0][0], "shares")  # shared moment ranks first


if __name__ == "__main__":
    unittest.main()
