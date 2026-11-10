import unittest

import numpy as np

from audio_similarity.preprocess.segment import count_segments, frame_signal


class TestFrameSignal(unittest.TestCase):
    sr = 100  # small sr keeps the tests fast and exact

    def test_exact_windows_with_overlap(self):
        # 20 s signal, 10 s window, 5 s hop -> starts at 0,5,10 (15 is a 5 s tail)
        sig = np.arange(20 * self.sr, dtype=np.float32)
        segs = frame_signal(sig, self.sr, 10, 5, 3)
        self.assertEqual(segs.shape[1], 10 * self.sr)
        # starts 0,5,10 are full; 15 is a 5 s padded tail (>= 3 s min)
        self.assertEqual(segs.shape[0], 4)
        # first window is the first 10 s verbatim
        np.testing.assert_array_equal(segs[0], sig[: 10 * self.sr])

    def test_tail_padded_to_full_window(self):
        # 13 s signal, window 10 s, hop 5 s: full window [0,10), padded tail [5,13).
        sig = np.ones(13 * self.sr, dtype=np.float32)
        segs = frame_signal(sig, self.sr, 10, 5, 3)
        self.assertEqual(segs.shape[0], 2)
        tail = segs[-1]
        self.assertEqual(tail.shape[0], 10 * self.sr)
        real = 8 * self.sr  # samples [5,13) = 8 s of real audio
        self.assertTrue(np.all(tail[:real] == 1.0))
        self.assertTrue(np.all(tail[real:] == 0.0))  # zero-padded remainder

    def test_short_tail_dropped(self):
        # min (7 s) > hop (5 s): the 6 s tail at start=5 falls below min and is dropped.
        sig = np.ones(11 * self.sr, dtype=np.float32)
        segs = frame_signal(sig, self.sr, 10, 5, 7)
        self.assertEqual(segs.shape[0], 1)

    def test_clip_shorter_than_window_is_kept(self):
        sig = np.ones(4 * self.sr, dtype=np.float32)
        segs = frame_signal(sig, self.sr, 10, 5, 3)
        self.assertEqual(segs.shape[0], 1)
        self.assertEqual(segs.shape[1], 10 * self.sr)

    def test_very_short_clip_still_yields_one_segment(self):
        sig = np.ones(1 * self.sr, dtype=np.float32)  # below min, but non-empty
        segs = frame_signal(sig, self.sr, 10, 5, 3)
        self.assertEqual(segs.shape[0], 1)

    def test_empty_signal(self):
        segs = frame_signal(np.array([], dtype=np.float32), self.sr, 10, 5, 3)
        self.assertEqual(segs.shape[0], 0)

    def test_no_pad_drops_tail(self):
        sig = np.ones(13 * self.sr, dtype=np.float32)
        segs = frame_signal(sig, self.sr, 10, 5, 3, pad_tail=False)
        self.assertEqual(segs.shape[0], 1)

    def test_count_matches_frame(self):
        for n in (0, 50, 400, 1300, 2000, 2500):
            self.assertEqual(
                count_segments(n, self.sr, 10, 5, 3),
                frame_signal(np.zeros(n, np.float32), self.sr, 10, 5, 3).shape[0],
            )


if __name__ == "__main__":
    unittest.main()
