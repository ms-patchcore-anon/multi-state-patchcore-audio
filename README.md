# Multi-State PatchCore Audio

Anonymous code submission for industrial audio anomaly detection.

Place the pretrained PANNs checkpoint here:
external/checkpoints/Cnn14_mAP=0.431.pth

PatchCore Setup:
raw waveform
→ pretrained CNN14 frontend
→ pretrained conv_block1–3
→ global average pooling
→ 256-dim embedding
→ optional L2 normalization
→ PatchCore memory bank
