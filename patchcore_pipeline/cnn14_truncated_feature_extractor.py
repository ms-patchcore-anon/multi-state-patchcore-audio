"""
cnn14_truncated_feature_extractor.py
------------------------------------

Feature extractor for short industrial audio windows based on pretrained
CNN14 from PANNs, truncated to conv_block1-3.

This module is used by the PatchCore-style pipeline for embedding extraction.

Important:
- No supervised fine-tuning is used.
- No classification head is used.
- No randomly initialized projection layer is used.
- Embeddings are extracted only from the pretrained CNN14 frontend and
  pretrained convolutional blocks.

Input:
- raw waveform tensor of shape (B, T)

Output:
- embedding tensor of shape (B, 256)
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from external.panns_inference.models import Cnn14


class CNN14TruncatedExtractor(nn.Module):
    def __init__(
        self,
        pann_ckpt_path: str | Path,
        freeze_frontend: bool = True,
        l2_normalize: bool = True,
    ):
        """
        Args:
            pann_ckpt_path:
                Path to the pretrained PANNs CNN14 checkpoint.
            freeze_frontend:
                If True, freeze the CNN14 frontend and convolutional blocks.
                This is the recommended setting for PatchCore-style inference.
            l2_normalize:
                If True, L2-normalize the extracted embeddings.
        """
        super().__init__()

        self.l2_normalize = l2_normalize

        pann_ckpt_path = Path(pann_ckpt_path).expanduser().resolve()
        if not pann_ckpt_path.exists():
            raise FileNotFoundError(f"Missing PANN checkpoint: {pann_ckpt_path}")

        # --------------------------------------------------
        # 1) Load pretrained CNN14
        # --------------------------------------------------
        cnn14 = Cnn14(
            sample_rate=25600,
            window_size=1024,
            hop_size=320,
            mel_bins=64,
            fmin=50,
            fmax=14000,
            classes_num=527,
        )

        pann_ckpt = torch.load(str(pann_ckpt_path), map_location="cpu")
        cnn14.load_state_dict(pann_ckpt["model"], strict=False)

        # --------------------------------------------------
        # 2) Reuse pretrained CNN14 frontend
        # --------------------------------------------------
        self.spectrogram_extractor = cnn14.spectrogram_extractor
        self.logmel_extractor = cnn14.logmel_extractor
        self.bn0 = cnn14.bn0

        # --------------------------------------------------
        # 3) Reuse pretrained early CNN14 convolution blocks
        # --------------------------------------------------
        self.conv_block1 = cnn14.conv_block1
        self.conv_block2 = cnn14.conv_block2
        self.conv_block3 = cnn14.conv_block3

        # --------------------------------------------------
        # 4) Freeze all parameters for inference-only extraction
        # --------------------------------------------------
        if freeze_frontend:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wave:
                Float tensor of shape (B, T).

        Returns:
            emb:
                Tensor of shape (B, 256).
        """

        # --------------------------------------------------
        # CNN14 frontend: waveform -> log-Mel representation
        # --------------------------------------------------
        x = self.spectrogram_extractor(wave)
        x = self.logmel_extractor(x)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        # --------------------------------------------------
        # Truncated pretrained CNN14 backbone
        # --------------------------------------------------
        x = self.conv_block1(x, pool_size=(1, 2), pool_type="avg")
        x = self.conv_block2(x, pool_size=(1, 2), pool_type="avg")
        x = self.conv_block3(x, pool_size=(1, 2), pool_type="avg")

        # --------------------------------------------------
        # Global average pooling
        # --------------------------------------------------
        emb = torch.mean(x, dim=(2, 3))  # (B, 256)

        # --------------------------------------------------
        # Optional L2 normalization
        # --------------------------------------------------
        if self.l2_normalize:
            emb = emb / (emb.norm(p=2, dim=1, keepdim=True) + 1e-9)

        return emb