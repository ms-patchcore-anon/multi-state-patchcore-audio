"""
cnn14_truncated_finetune.py
---------------------------

Truncated CNN14 model for short industrial audio classification.

This model reuses the pretrained CNN14 frontend and the first three
convolutional blocks from PANNs/CNN14, followed by a lightweight embedding
head and classification head.

The model is intended for short waveform windows, such as 0.2 s and 0.4 s
segments, where the full CNN14 backbone is too deep for stable short-context
representations.

Architecture:
- Raw waveform input
- CNN14 frontend:
    STFT -> Mel -> log-Mel
- CNN14 convolution blocks:
    conv_block1
    conv_block2
    conv_block3
- Global average pooling
- Linear embedding layer
- Classification head

Notes:
- No energy-fusion feature is used in this version.
- The frontend can remain frozen by default.
- The first three convolutional blocks and the classification heads are trainable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from external.panns_inference.models import Cnn14


class CNN14TruncatedFineTune(nn.Module):
    """
    Truncated CNN14 classifier for short waveform inputs.

    Args:
        checkpoint_path:
            Path to the pretrained CNN14 checkpoint.
        num_classes:
            Number of target classes.
        emb_dim:
            Dimension of the learned embedding before classification.
        freeze_frontend:
            If True, keep the spectrogram/log-Mel frontend frozen.
    """

    def __init__(
        self,
        checkpoint_path,
        num_classes: int = 3,
        emb_dim: int = 128,
        freeze_frontend: bool = True,
    ):
        super().__init__()

        # --------------------------------------------------
        # 1) Load pretrained CNN14
        # --------------------------------------------------
        self.cnn14 = Cnn14(
            sample_rate=25600,
            window_size=1024,
            hop_size=320,
            mel_bins=64,
            fmin=50,
            fmax=14000,
            classes_num=527,
        )

        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        self.cnn14.load_state_dict(checkpoint["model"], strict=False)

        # --------------------------------------------------
        # 2) Reuse CNN14 frontend
        # --------------------------------------------------
        self.spectrogram_extractor = self.cnn14.spectrogram_extractor
        self.logmel_extractor = self.cnn14.logmel_extractor
        self.bn0 = self.cnn14.bn0

        # --------------------------------------------------
        # 3) Use only early CNN14 convolution blocks
        # --------------------------------------------------
        self.conv_block1 = self.cnn14.conv_block1
        self.conv_block2 = self.cnn14.conv_block2
        self.conv_block3 = self.cnn14.conv_block3

        # --------------------------------------------------
        # 4) Lightweight classification head
        # --------------------------------------------------
        self.fc_emb = nn.Linear(256, emb_dim)
        self.classifier = nn.Linear(emb_dim, num_classes)

        # --------------------------------------------------
        # 5) Freeze all parameters first
        # --------------------------------------------------
        for p in self.parameters():
            p.requires_grad = False

        # --------------------------------------------------
        # 6) Train embedding and classifier heads
        # --------------------------------------------------
        for p in self.fc_emb.parameters():
            p.requires_grad = True

        for p in self.classifier.parameters():
            p.requires_grad = True

        # --------------------------------------------------
        # 7) Fine-tune truncated convolutional backbone
        # --------------------------------------------------
        for p in self.conv_block1.parameters():
            p.requires_grad = True

        for p in self.conv_block2.parameters():
            p.requires_grad = True

        for p in self.conv_block3.parameters():
            p.requires_grad = True

        # --------------------------------------------------
        # 8) Optionally unfreeze frontend
        # --------------------------------------------------
        if not freeze_frontend:
            for p in self.spectrogram_extractor.parameters():
                p.requires_grad = True

            for p in self.logmel_extractor.parameters():
                p.requires_grad = True

            for p in self.bn0.parameters():
                p.requires_grad = True

    def forward(self, wave):
        """
        Forward pass.

        Args:
            wave:
                Tensor of shape (B, T), where B is batch size and T is the
                number of waveform samples.

        Returns:
            logits:
                Tensor of shape (B, num_classes).
            emb:
                Tensor of shape (B, emb_dim).
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
        # Truncated CNN14 backbone
        # --------------------------------------------------
        x = self.conv_block1(x, pool_size=(1, 2), pool_type="avg")
        x = self.conv_block2(x, pool_size=(1, 2), pool_type="avg")
        x = self.conv_block3(x, pool_size=(1, 2), pool_type="avg")

        # --------------------------------------------------
        # Global pooling and classification
        # --------------------------------------------------
        x = torch.mean(x, dim=(2, 3))

        emb = F.relu(self.fc_emb(x))
        logits = self.classifier(emb)

        return logits, emb