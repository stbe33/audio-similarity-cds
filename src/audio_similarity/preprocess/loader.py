"""Decode an audio file to a mono float32 waveform at a target sample rate.

``librosa`` is imported lazily so importing the package (e.g. for the pure-logic
tests) does not require the audio stack to be installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_audio(
    path: str | Path, target_sr: int, mono: bool = True
) -> np.ndarray:
    """Load ``path`` as a 1-D float32 array resampled to ``target_sr``."""
    import librosa  # lazy: heavy, optional at import time

    y, _ = librosa.load(str(path), sr=target_sr, mono=mono)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:  # safety: collapse any residual channel dim
        y = y.mean(axis=0)
    return y
