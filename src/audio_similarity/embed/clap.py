"""LAION-CLAP encoder (``laion/larger_clap_music``) via HuggingFace Transformers.

CLAP expects 48 kHz mono waveforms and projects audio (and text) into a shared
512-d space, so the same model can answer both "songs like this song" and
"songs like this text description". torch/transformers are imported lazily.
"""

from __future__ import annotations

import numpy as np

from .base import Embedder, iter_batches, resolve_device


class ClapEmbedder(Embedder):
    name = "clap"
    target_sr = 48000  # required by the CLAP feature extractor

    def __init__(
        self,
        checkpoint: str = "laion/larger_clap_music",
        device: str = "auto",
        batch_size: int = 16,
    ) -> None:
        import torch
        from transformers import ClapModel, ClapProcessor

        self._torch = torch
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self.device = resolve_device(device)

        self.model = ClapModel.from_pretrained(checkpoint).to(self.device).eval()
        self.processor = ClapProcessor.from_pretrained(checkpoint)
        self.dim = int(self.model.config.projection_dim)

    def embed_segments(self, segments: np.ndarray) -> np.ndarray:
        segments = np.asarray(segments, dtype=np.float32)
        if segments.ndim != 2:
            raise ValueError(f"expected (n, samples), got {segments.shape}")

        outputs = []
        for batch in iter_batches(segments, self.batch_size):
            inputs = self.processor(
                audios=list(batch),
                sampling_rate=self.target_sr,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with self._torch.no_grad():
                feats = self.model.get_audio_features(**inputs)
            outputs.append(feats.detach().cpu().float().numpy())
        return np.concatenate(outputs, axis=0).astype(np.float32)

    def embed_text(self, texts: list[str]) -> np.ndarray:
        inputs = self.processor(text=list(texts), return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self._torch.no_grad():
            feats = self.model.get_text_features(**inputs)
        return feats.detach().cpu().float().numpy().astype(np.float32)
