import unittest

import numpy as np

from audio_similarity.aggregate import aggregate_segments, l2_normalize, mean_pool


class TestAggregate(unittest.TestCase):
    def test_l2_normalize_unit_length(self):
        x = np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)
        n = l2_normalize(x, axis=1)
        np.testing.assert_allclose(np.linalg.norm(n, axis=1), [1.0, 1.0], atol=1e-6)

    def test_l2_normalize_zero_vector_safe(self):
        z = np.zeros(4, dtype=np.float32)
        out = l2_normalize(z)
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertAlmostEqual(float(np.linalg.norm(out)), 0.0, places=5)

    def test_mean_pool(self):
        sv = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        np.testing.assert_allclose(mean_pool(sv), [2.0, 3.0])

    def test_mean_pool_rejects_empty(self):
        with self.assertRaises(ValueError):
            mean_pool(np.zeros((0, 5), dtype=np.float32))

    def test_aggregate_is_unit_norm(self):
        sv = np.random.default_rng(0).normal(size=(7, 16)).astype(np.float32)
        song = aggregate_segments(sv, normalize=True)
        self.assertEqual(song.shape, (16,))
        self.assertAlmostEqual(float(np.linalg.norm(song)), 1.0, places=5)

    def test_aggregate_without_norm_is_plain_mean(self):
        sv = np.array([[1.0, 1.0], [3.0, 3.0]], dtype=np.float32)
        np.testing.assert_allclose(aggregate_segments(sv, normalize=False), [2.0, 2.0])


if __name__ == "__main__":
    unittest.main()
