"""
train_cnn14_stage_unfreeze_weightedloss.py
------------------------------------------

Train a truncated CNN14 classifier for short industrial audio waveform windows.

This script trains the supervised CNN baseline used for comparison with the
PatchCore-style retrieval pipeline.

The model uses:
- pretrained CNN14 frontend from PANNs
- truncated CNN14 convolutional backbone, conv_block1 to conv_block3
- lightweight embedding layer
- classification head

Default paths point to the sample 0.2 s output structure used in this repository.
For 0.4 s windows or private datasets, pass paths via command-line arguments.

Expected metadata:
- train waveform metadata CSV
- validation waveform metadata CSV

Each metadata CSV is expected to contain at least:
- path
- label

The path column should point directly to extracted .wav windows.

Example usage for 0.2 s windows:
    python classification_pipeline/train_cnn14_stage_unfreeze_weightedloss.py \
        --train_csv outputs/sample_windows_0p2s_3class/File_split_CSVs/train_metadata_waveform.csv \
        --val_csv outputs/sample_windows_0p2s_3class/File_split_CSVs/val_metadata_waveform.csv \
        --pann_checkpoint external/checkpoints/Cnn14_mAP=0.431.pth \
        --output_dir outputs/classification_0p2s

Example usage for 0.4 s windows:
    python classification_pipeline/train_cnn14_stage_unfreeze_weightedloss.py \
        --train_csv outputs/sample_windows_0p4s_3class/File_split_CSVs/train_metadata_waveform.csv \
        --val_csv outputs/sample_windows_0p4s_3class/File_split_CSVs/val_metadata_waveform.csv \
        --pann_checkpoint external/checkpoints/Cnn14_mAP=0.431.pth \
        --output_dir outputs/classification_0p4s
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import argparse
import platform
import os
import random

import numpy as np

# =========================================================
# REPRODUCIBILITY
# IMPORTANT: set environment variables before importing torch
# =========================================================
SEED = 0

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["PYTHONHASHSEED"] = str(SEED)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from waveform_dataset import WaveformDataset
from cnn14_truncated_finetune import CNN14TruncatedFineTune


# =========================================================
# ARGUMENTS
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train truncated CNN14 classifier for short waveform windows."
    )

    parser.add_argument(
        "--train_csv",
        type=Path,
        default=Path(
            "outputs/sample_windows_0p2s_3class/File_split_CSVs/"
            "train_metadata_waveform.csv"
        ),
        help="Path to training waveform metadata CSV.",
    )

    parser.add_argument(
        "--val_csv",
        type=Path,
        default=Path(
            "outputs/sample_windows_0p2s_3class/File_split_CSVs/"
            "val_metadata_waveform.csv"
        ),
        help="Path to validation waveform metadata CSV.",
    )

    parser.add_argument(
        "--pann_checkpoint",
        type=Path,
        default=Path("external/checkpoints/Cnn14_mAP=0.431.pth"),
        help="Path to pretrained PANNs CNN14 checkpoint.",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/classification_0p2s"),
        help="Directory where trained model files will be saved.",
    )

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=25)

    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--lr_block3", type=float, default=5e-6)

    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()


# =========================================================
# REPRODUCIBILITY HELPERS
# =========================================================
def set_global_seed(seed: int = 0, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.set_num_threads(1)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass


def seed_worker(worker_id: int) -> None:
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


# =========================================================
# TRAINABLE CONTROL
# =========================================================
def set_trainable(module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    args = parse_args()

    global SEED
    SEED = int(args.seed)

    set_global_seed(SEED, deterministic=True)

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_csv = args.train_csv
    val_csv = args.val_csv
    pann_checkpoint = args.pann_checkpoint
    output_dir = args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = output_dir / "cnn14_stage_unfreeze_weighted_best.pth"
    last_model_path = output_dir / "cnn14_stage_unfreeze_weighted_last.pth"

    batch_size = int(args.batch_size)
    num_workers = int(args.num_workers)
    epochs = int(args.epochs)

    lr_head = float(args.lr_head)
    lr_block3 = float(args.lr_block3)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    assert train_csv.exists(), f"Train CSV not found: {train_csv}"
    assert val_csv.exists(), f"Val CSV not found: {val_csv}"
    assert pann_checkpoint.exists(), f"PANN checkpoint not found: {pann_checkpoint}"

    # =========================================================
    # RUN INFO
    # =========================================================
    print("=== RUN INFO ===")
    print("time:           ", datetime.now().isoformat())
    print("seed:           ", SEED)
    print("python:         ", platform.python_version())
    print("torch:          ", torch.__version__)
    print("cuda available: ", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:            ", torch.cuda.get_device_name(0))
    print("train_csv:      ", train_csv)
    print("val_csv:        ", val_csv)
    print("pann ckpt:      ", pann_checkpoint)
    print("output_dir:     ", output_dir)
    print("best save path: ", best_model_path)
    print("last save path: ", last_model_path)
    print("================")

    # =========================================================
    # DATA
    # =========================================================
    train_ds = WaveformDataset(train_csv, sample_rate=25600)
    val_ds = WaveformDataset(val_csv, sample_rate=25600)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    print(f"[DATA] Train samples: {len(train_ds)}")
    print(f"[DATA] Val samples:   {len(val_ds)}")

    # =========================================================
    # MODEL
    # =========================================================
    model = CNN14TruncatedFineTune(
        checkpoint_path=pann_checkpoint,
        num_classes=3,
        emb_dim=128,
        freeze_frontend=True,
    ).to(device)

    # =========================================================
    # STAGE CONTROL
    # =========================================================
    # Stage 1: train head only
    set_trainable(model.conv_block1, False)
    set_trainable(model.conv_block2, False)
    set_trainable(model.conv_block3, False)
    set_trainable(model.fc_emb, True)
    set_trainable(model.classifier, True)

    print("[STAGE] Epochs 1-5: Head only, conv blocks frozen")

    # =========================================================
    # LOSS
    # =========================================================
    class_weights = torch.tensor([1.0, 1.2, 1.5], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # =========================================================
    # OPTIMIZER
    # =========================================================
    def build_optimizer(stage: str):
        params = [
            {"params": model.fc_emb.parameters(), "lr": lr_head},
            {"params": model.classifier.parameters(), "lr": lr_head},
        ]

        if stage == "block3":
            params += [{"params": model.conv_block3.parameters(), "lr": lr_block3}]

        return torch.optim.Adam(params)

    optimizer = build_optimizer(stage="head")

    # =========================================================
    # TRAIN / VAL LOOP
    # =========================================================
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        print(f"\n=== Epoch {epoch}/{epochs} ===")

        # Unfreeze conv_block3 at epoch 6
        if epoch == 6:
            set_trainable(model.conv_block3, True)
            optimizer = build_optimizer(stage="block3")
            print("[STAGE] Epochs 6+: Unfreeze conv_block3 with small LR")

        # ---------------- TRAIN ----------------
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for wave, label in train_loader:
            wave = wave.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            logits, _ = model(wave)
            loss = criterion(logits, label)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * wave.size(0)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == label).sum().item()
            train_total += label.size(0)

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total
        print(f"[TRAIN] loss={train_loss:.4f}, acc={train_acc:.4f}")

        # ---------------- VAL ----------------
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for wave, label in val_loader:
                wave = wave.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)

                logits, _ = model(wave)
                loss = criterion(logits, label)

                val_loss_sum += loss.item() * wave.size(0)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == label).sum().item()
                val_total += label.size(0)

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total
        print(f"[VAL]   loss={val_loss:.4f}, acc={val_acc:.4f}")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(
                f"[SAVE] New best model saved: {best_model_path} "
                f"(val_acc={val_acc:.4f})"
            )

        # Always save last
        torch.save(model.state_dict(), last_model_path)

    print("\nDone.")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best model path: {best_model_path}")
    print(f"Last model path: {last_model_path}")


if __name__ == "__main__":
    main()