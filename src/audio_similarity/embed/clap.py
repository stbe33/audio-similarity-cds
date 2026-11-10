"""LAION-CLAP encoder (``laion/larger_clap_music``) via HuggingFace Transformers.

CLAP expects 48 kHz mono waveforms and projects audio (and text) into a shared
512-d space, so the same model can answer both "songs like this song" and
"songs like this text description". torch/transformers are imported lazily.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .base import Embedder, iter_batches, resolve_device

# CLAP's audio frontend uses ops (e.g. bicubic upsample) that some torch/MPS
# builds lack; let unsupported MPS ops fall back to CPU instead of crashing.
# Set before torch is imported (clap.py loads before ClapEmbedder.__init__).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _prepare_checkpoint(checkpoint: str) -> str:
    """Return a local path transformers can load, converting .bin -> safetensors.

    transformers refuses to ``torch.load`` pickle (.bin) weights on torch < 2.6
    (CVE-2025-32434). Several LAION CLAP checkpoints (incl. ``larger_clap_music``)
    ship only ``.bin``, so for a hub id we fetch the (cached) snapshot and, if it
    has no safetensors, create one via torch's SAFE ``weights_only=True`` load.
    This keeps embedding working on torch 2.2 without forcing an upgrade. A local
    path, or anything we cannot convert, is returned unchanged for transformers
    to handle (or error on) directly.
    """
    path = Path(checkpoint)
    if path.exists():
        return str(checkpoint)
    try:
        from huggingface_hub import snapshot_download

        snap = Path(snapshot_download(checkpoint))
    except Exception:
        return checkpoint  # offline / non-hub id: let from_pretrained decide

    safetensors = snap / "model.safetensors"
    pickle_bin = snap / "pytorch_model.bin"
    if not safetensors.exists() and pickle_bin.exists():
        try:
            import torch
            from safetensors.torch import save_file

            state = torch.load(pickle_bin, map_location="cpu", weights_only=True)
            state = {
                k: v.contiguous().clone()
                for k, v in state.items()
                if isinstance(v, torch.Tensor)
            }
            save_file(state, str(safetensors), metadata={"format": "pt"})
        except Exception:
            return checkpoint
    return str(snap)


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

        resolved = _prepare_checkpoint(checkpoint)
        self.model = ClapModel.from_pretrained(resolved).to(self.device).eval()
        self.processor = ClapProcessor.from_pretrained(resolved)
        self.dim = int(self.model.config.projection_dim)

    def _fallback_to_cpu(self) -> None:
        """Move the model to CPU after an unsupported-accelerator op."""
        print(f"[clap] '{self.device}' is missing an op for CLAP; "
              "falling back to CPU for this model.")
        self.model = self.model.to("cpu")
        self.device = "cpu"

    def _audio_features(self, inputs: dict):
        moved = {k: v.to(self.device) for k, v in inputs.items()}
        with self._torch.no_grad():
            feats = self.model.get_audio_features(**moved)
        return feats.detach().cpu().float().numpy()

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
            try:
                feats = self._audio_features(inputs)
            except NotImplementedError:
                # e.g. an op not implemented on MPS and no CPU fallback kicked in.
                if self.device == "cpu":
                    raise
                self._fallback_to_cpu()
                feats = self._audio_features(inputs)
            outputs.append(feats)
        return np.concatenate(outputs, axis=0).astype(np.float32)

    def embed_text(self, texts: list[str]) -> np.ndarray:
        inputs = self.processor(text=list(texts), return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self._torch.no_grad():
            feats = self.model.get_text_features(**inputs)
        return feats.detach().cpu().float().numpy().astype(np.float32)
