import tempfile
import unittest
from pathlib import Path

from audio_similarity.data.metadata import (
    all_categories,
    load_autotagging_tsv,
)

TSV = (
    "TRACK_ID\tARTIST_ID\tALBUM_ID\tPATH\tDURATION\tTAGS\n"
    "track_1\ta1\talb1\t56/1376256.mp3\t180.0\tgenre---rock\tinstrument---guitar\n"
    "track_2\ta2\talb2\t12/42.mp3\t90.5\tgenre---jazz\tmood/theme---happy\n"
    "track_3\ta3\talb3\t00/7.mp3\tnotanumber\tgenre---rock\n"
)


class TestMetadata(unittest.TestCase):
    def _load(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "autotagging.tsv"
            p.write_text(TSV, encoding="utf-8")
            return load_autotagging_tsv(p)

    def test_keyed_by_stem(self):
        tracks = self._load()
        self.assertEqual(set(tracks), {"1376256", "42", "7"})

    def test_fields_parsed(self):
        tracks = self._load()
        m = tracks["1376256"]
        self.assertEqual(m.track_id, "track_1")
        self.assertEqual(m.path, "56/1376256.mp3")
        self.assertAlmostEqual(m.duration, 180.0)
        self.assertEqual(m.tags, frozenset({"genre---rock", "instrument---guitar"}))

    def test_tags_in_category(self):
        tracks = self._load()
        self.assertEqual(tracks["42"].tags_in("mood/theme"), {"mood/theme---happy"})
        self.assertEqual(tracks["42"].tags_in("instrument"), set())

    def test_bad_duration_defaults_zero(self):
        tracks = self._load()
        self.assertEqual(tracks["7"].duration, 0.0)

    def test_all_categories(self):
        self.assertEqual(
            all_categories(self._load()), {"genre", "instrument", "mood/theme"}
        )


if __name__ == "__main__":
    unittest.main()
