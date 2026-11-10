import unittest

from audio_similarity.evaluate.metrics import (
    average_jaccard_at_k,
    dcg,
    filter_category,
    jaccard,
    ndcg_at_k,
    precision_at_k,
    shares_tag,
)


class TestMetrics(unittest.TestCase):
    def test_jaccard(self):
        self.assertAlmostEqual(jaccard({"a", "b"}, {"b", "c"}), 1 / 3)
        self.assertEqual(jaccard(set(), set()), 0.0)
        self.assertEqual(jaccard({"a"}, {"a"}), 1.0)

    def test_shares_tag(self):
        self.assertTrue(shares_tag({"genre---rock"}, {"genre---rock", "genre---pop"}))
        self.assertFalse(shares_tag({"genre---rock"}, {"genre---jazz"}))

    def test_filter_category(self):
        tags = {"genre---rock", "genre---pop", "mood/theme---happy", "instrument---guitar"}
        self.assertEqual(filter_category(tags, "genre"), {"genre---rock", "genre---pop"})
        self.assertEqual(filter_category(tags, "mood/theme"), {"mood/theme---happy"})

    def test_precision_at_k(self):
        q = {"genre---rock"}
        ranked = [{"genre---rock"}, {"genre---jazz"}, {"genre---rock"}]
        self.assertAlmostEqual(precision_at_k(ranked, q, 3), 2 / 3)
        self.assertAlmostEqual(precision_at_k(ranked, q, 1), 1.0)

    def test_precision_k_larger_than_list(self):
        self.assertAlmostEqual(precision_at_k([{"a"}], {"a"}, 10), 1.0)

    def test_average_jaccard_at_k(self):
        q = {"a", "b"}
        ranked = [{"a", "b"}, {"a", "c"}]
        # jaccard = 1.0 and 1/3 -> mean 2/3
        self.assertAlmostEqual(average_jaccard_at_k(ranked, q, 2), (1.0 + 1 / 3) / 2)

    def test_dcg_monotonic_in_relevance(self):
        self.assertGreater(dcg([3, 2, 1]), dcg([1, 2, 3]))

    def test_ndcg_perfect_ranking_is_one(self):
        rels = [3, 2, 1]
        self.assertAlmostEqual(ndcg_at_k(rels, 3, ideal_relevances=rels), 1.0)

    def test_ndcg_worst_ranking_below_perfect(self):
        pool = [3, 2, 1]
        worst = ndcg_at_k([1, 2, 3], 3, ideal_relevances=pool)
        self.assertLess(worst, 1.0)
        self.assertGreater(worst, 0.0)

    def test_ndcg_all_zero_relevance(self):
        self.assertEqual(ndcg_at_k([0, 0, 0], 3, ideal_relevances=[0, 0, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
