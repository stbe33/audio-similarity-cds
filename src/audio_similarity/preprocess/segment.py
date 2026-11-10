"""Slice a 1-D waveform into fixed-length (optionally overlapping) windows.

Design choices (see PLAN.md):
- Windows have a fixed length so the encoder sees uniform inputs.
- Tails at least ``min_segment_seconds`` long are zero-padded to a full window
  rather than dropped, so the end of a track still contributes.
- Any non-empty signal yields at least one segment (a very short clip is padded).
"""

from __future__ import annotations

import numpy as np


def frame_signal(
    signal: np.ndarray,
    sr: int,
    window_seconds: float = 10.0,
    hop_seconds: float = 5.0,
    min_segment_seconds: float = 3.0,
    pad_tail: bool = True,
) -> np.ndarray:
    """Return a ``(n_segments, window_samples)`` float32 array.

    Parameters
    ----------
    signal : 1-D array of samples (mono).
    sr : sample rate of ``signal``.
    window_seconds, hop_seconds : window length and stride in seconds.
    min_segment_seconds : a trailing window shorter than this is dropped,
        unless it is the only candidate (then it is padded so the clip is kept).
    pad_tail : if False, short tails are dropped and never padded.
    """
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    n = signal.shape[0]

    window = max(1, int(round(window_seconds * sr)))
    hop = max(1, int(round(hop_seconds * sr)))
    min_samples = max(1, int(round(min_segment_seconds * sr)))

    if n == 0:
        return np.zeros((0, window), dtype=np.float32)

    segments: list[np.ndarray] = []
    start = 0
    while start < n:
        end = start + window
        if end <= n:
            segments.append(signal[start:end])
            start += hop
            continue

        # Partial trailing window.
        remaining = n - start
        if pad_tail and remaining >= min_samples:
            seg = np.zeros(window, dtype=np.float32)
            seg[:remaining] = signal[start:]
            segments.append(seg)
        break

    # Signal shorter than one window, or too short to clear the min-tail rule:
    # keep the clip by padding a single window so nothing yields zero segments.
    if not segments:
        if not pad_tail:
            return np.zeros((0, window), dtype=np.float32)
        seg = np.zeros(window, dtype=np.float32)
        seg[:n] = signal
        segments.append(seg)

    return np.stack(segments).astype(np.float32)


def count_segments(
    n_samples: int,
    sr: int,
    window_seconds: float = 10.0,
    hop_seconds: float = 5.0,
    min_segment_seconds: float = 3.0,
) -> int:
    """Number of segments :func:`frame_signal` would produce (without allocating)."""
    dummy = np.zeros(n_samples, dtype=np.float32)
    return frame_signal(
        dummy, sr, window_seconds, hop_seconds, min_segment_seconds
    ).shape[0]
