import unittest

from audio_similarity.data.metadata import TrackMeta
from audio_similarity.data.subset import stratified_sample


def make_tracks():
    # 30 rock, 30 jazz, 6 pop -> deliberately imbalanced
    tracks = {}
    plan = [("rock", 30), ("jazz", 30), ("pop", 6)]
    idx = 0
    for genre, count in plan:
        for _ in range(count):
            key = f"t{idx}"
            tracks[key] = TrackMeta(
                track_id=key, path=f"00/{idx}.mp3", duration=10.0,
                tags=frozenset({f"genre---{genre}"}),
            )
            idx += 1
    return tracks


class TestSubset(unittest.TestCase):
    def test_returns_n_distinct(self):
        keys = stratified_sample(make_tracks(), 24, seed=0)
        self.assertEqual(len(keys), 24)
        self.assertEqual(len(set(keys)), 24)

    def test_balances_across_genres(self):
        tracks = make_tracks()
        keys = stratified_sample(tracks, 18, seed=0)
        genres = [next(iter(tracks[k].tags)).split("---")[1] for k in keys]
        # Round-robin should take evenly until the small genre runs out:
        # 6 pop available, so pop should contribute all 6; rock/jazz 6 each.
        self.assertEqual(genres.count("pop"), 6)
        self.assertEqual(genres.count("rock"), 6)
        self.assertEqual(genres.count("jazz"), 6)

    def test_deterministic_with_seed(self):
        t = make_tracks()
        self.assertEqual(stratified_sample(t, 20, seed=7), stratified_sample(t, 20, seed=7))

    def test_caps_at_total(self):
        keys = stratified_sample(make_tracks(), 10_000, seed=1)
        self.assertEqual(len(keys), 66)  # 30 + 30 + 6


if __name__ == "__main__":
    unittest.main()
