import tempfile
import unittest
from pathlib import Path

# Importing clap.py must not require torch/transformers (they load lazily).
from audio_similarity.embed.clap import _prepare_checkpoint


class TestPrepareCheckpoint(unittest.TestCase):
    def test_local_path_is_returned_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(_prepare_checkpoint(d), d)

    def test_local_path_with_trailing_structure(self):
        with tempfile.TemporaryDirectory() as d:
            sub = Path(d) / "model_dir"
            sub.mkdir()
            self.assertEqual(_prepare_checkpoint(str(sub)), str(sub))


if __name__ == "__main__":
    unittest.main()
