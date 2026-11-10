import tempfile
import unittest
from pathlib import Path

import numpy as np

from audio_similarity.aggregate import l2_normalize
from audio_similarity.index.search import cosine_topk, rank_all
from audio_similarity.index.store import load_index, save_index


class TestSearch(unittest.TestCase):
    def setUp(self):
        # Four 2-D unit vectors at distinct angles.
        self.matrix = l2_normalize(
            np.array([[1, 0], [0.9, 0.1], [0, 1], [-1, 0]], dtype=np.float32), axis=1
        )

    def test_topk_orders_by_similarity(self):
        q = self.matrix[0]  # closest to row 0, then row 1
        idx, scores = cosine_topk(q, self.matrix, k=2, exclude_index=0)
        self.assertEqual(list(idx), [1, 2])  # row 1 nearest after excluding self
        self.assertTrue(scores[0] >= scores[1])

    def test_exclude_self_removes_query_row(self):
        idx, _ = cosine_topk(self.matrix[2], self.matrix, k=4, exclude_index=2)
        self.assertNotIn(2, list(idx))

    def test_k_clamped_to_available(self):
        idx, scores = cosine_topk(self.matrix[0], self.matrix, k=99, exclude_index=0)
        self.assertEqual(len(idx), 3)

    def test_rank_all_is_full_permutation(self):
        order, _ = rank_all(self.matrix[0], self.matrix, exclude_index=0)
        self.assertEqual(sorted(order.tolist()), [1, 2, 3])

    def test_assume_normalized_false_matches_true(self):
        raw = np.array([[2, 0], [0, 5], [3, 3]], dtype=np.float32)
        q = np.array([4, 0], dtype=np.float32)
        idx_raw, _ = cosine_topk(q, raw, k=3, assume_normalized=False)
        idx_norm, _ = cosine_topk(
            l2_normalize(q), l2_normalize(raw, axis=1), k=3, assume_normalized=True
        )
        self.assertEqual(list(idx_raw), list(idx_norm))


class TestStore(unittest.TestCase):
    def test_roundtrip(self):
        ids = ["1376256", "42", "track_x"]
        vecs = np.random.default_rng(1).normal(size=(3, 8)).astype(np.float32)
        with tempfile.TemporaryDirectory() as d:
            p = save_index(Path(d) / "idx.npz", ids, vecs)
            got_ids, got_vecs = load_index(p)
        self.assertEqual(got_ids, ids)
        np.testing.assert_allclose(got_vecs, vecs, atol=1e-6)

    def test_length_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                save_index(Path(d) / "idx.npz", ["a"], np.zeros((2, 4), np.float32))


if __name__ == "__main__":
    unittest.main()
