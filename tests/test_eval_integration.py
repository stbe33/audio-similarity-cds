"""End-to-end check of the retrieval + evaluation path with synthetic data.

Embeddings are built to encode genre (a one-hot direction plus small noise), so a
cosine ranking must place same-genre tracks first and clear the random floor.
This exercises store -> search -> evaluate without needing torch or audio.
"""

import unittest

import numpy as np

from audio_similarity.aggregate import l2_normalize
from audio_similarity.data.metadata import TrackMeta
from audio_similarity.evaluate.run_eval import evaluate_index, format_markdown


def build_synthetic(n_genres=4, per_genre=12, dim=16, noise=0.15, seed=0):
    rng = np.random.default_rng(seed)
    ids, vectors, tracks = [], [], {}
    centers = np.eye(n_genres, dim, dtype=np.float32)
    genres = ["rock", "jazz", "pop", "metal", "folk", "soul"][:n_genres]
    k = 0
    for gi, genre in enumerate(genres):
        for _ in range(per_genre):
            key = f"t{k}"
            vec = centers[gi] + noise * rng.standard_normal(dim).astype(np.float32)
            ids.append(key)
            vectors.append(vec)
            tracks[key] = TrackMeta(
                track_id=key, path=f"00/{k}.mp3", duration=10.0,
                tags=frozenset({f"genre---{genre}"}),
            )
            k += 1
    return ids, l2_normalize(np.stack(vectors), axis=1), tracks


class TestEvalIntegration(unittest.TestCase):
    def test_model_beats_random_floor(self):
        ids, vectors, tracks = build_synthetic()
        report = evaluate_index(
            ids, vectors, tracks, k=5, categories=["genre"], seed=1
        )
        model_p = report["model"]["genre"]["precision_at_k"]
        floor_p = report["random_floor"]["genre"]["precision_at_k"]
        # Clean genre structure -> near-perfect retrieval, well above chance (~0.25).
        self.assertGreater(model_p, 0.9)
        self.assertGreater(model_p, floor_p + 0.3)
        self.assertGreater(report["model"]["genre"]["ndcg_at_k"], 0.9)

    def test_report_structure_and_markdown(self):
        ids, vectors, tracks = build_synthetic()
        report = evaluate_index(ids, vectors, tracks, k=5, categories=["genre"])
        self.assertEqual(report["k"], 5)
        self.assertEqual(report["n_tracks"], len(ids))
        self.assertIn("all", report["model"])
        md = format_markdown(report)
        self.assertIn("Retrieval evaluation", md)
        self.assertIn("genre", md)

    def test_requires_two_tracks(self):
        ids, vectors, tracks = build_synthetic(n_genres=1, per_genre=1)
        with self.assertRaises(ValueError):
            evaluate_index(ids, vectors, tracks, k=5)


if __name__ == "__main__":
    unittest.main()
