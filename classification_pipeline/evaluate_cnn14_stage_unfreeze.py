"""
evaluate_cnn14_stage_unfreeze.py
------------------------------------------------

Evaluate a truncated CNN14 classifier on the test split.

Reports:
1) Window-level metrics:
   - Accuracy
   - Macro-F1
   - FAR
   - Macro ROC-AUC
   - Macro PR-AUC
   - Confusion matrix
   - Classification report

2) Event/file-level metrics using Top-K mean aggregation grouped by file_id:
   - Accuracy
   - Macro-F1
   - FAR
   - Macro ROC-AUC
   - Macro PR-AUC
   - Confusion matrix
   - Classification report

Event-level Top-K mean aggregation:
- For each file_id, collect all window-level softmax probabilities.
- Compute a confidence score per window as max softmax probability.
- Select the Top-K most confident windows for that file_id.
- Average their probability vectors.
- Final event prediction = argmax(mean_topk_probability).

FAR definition:
- Class_2 is treated as the malfunction class.
- Class_1 and Class_4 are treated as normal/background classes.
- FAR = normal/background samples predicted as malfunction / all normal/background samples.

This script uses waveform metadata directly. The metadata CSV is expected to
contain at least:
- path
- label
- file_id

The path column should point to extracted .wav windows.

Example usage for 0.2 s windows:
    python classification_pipeline/evaluate_cnn14_stage_unfreeze_eventlevel_topk.py \
        --test_csv outputs/sample_windows_0p2s_3class/File_split_CSVs/test_metadata_waveform.csv \
        --pann_checkpoint external/checkpoints/Cnn14_mAP=0.431.pth \
        --model_weights outputs/classification_0p2s/cnn14_stage_unfreeze_weighted_best.pth \
        --out_dir outputs/classification_0p2s/evaluation \
        --topk 5

Example usage for 0.4 s windows:
    python classification_pipeline/evaluate_cnn14_stage_unfreeze_eventlevel_topk.py \
        --test_csv outputs/sample_windows_0p4s_3class/File_split_CSVs/test_metadata_waveform.csv \
        --pann_checkpoint external/checkpoints/Cnn14_mAP=0.431.pth \
        --model_weights outputs/classification_0p4s/cnn14_stage_unfreeze_weighted_best.pth \
        --out_dir outputs/classification_0p4s/evaluation \
        --topk 5
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import argparse
import json
import os
import platform
import random
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize

from waveform_dataset import WaveformDataset
from cnn14_truncated_finetune import CNN14TruncatedFineTune


# =========================================================
# ARGUMENTS
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate truncated CNN14 at window level and event level "
            "using Top-K mean aggregation."
        )
    )

    parser.add_argument(
        "--test_csv",
        type=Path,
        default=Path(
            "outputs/sample_windows_0p2s_3class/File_split_CSVs/"
            "test_metadata_waveform.csv"
        ),
        help="Path to test waveform metadata CSV.",
    )

    parser.add_argument(
        "--pann_checkpoint",
        type=Path,
        default=Path("external/checkpoints/Cnn14_mAP=0.431.pth"),
        help="Path to pretrained PANNs CNN14 checkpoint.",
    )

    parser.add_argument(
        "--model_weights",
        type=Path,
        default=Path(
            "outputs/classification_0p2s/"
            "cnn14_stage_unfreeze_weighted_best.pth"
        ),
        help="Path to trained classifier weights.",
    )

    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("outputs/classification_0p2s/evaluation"),
        help="Directory for evaluation outputs.",
    )

    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", type=int, default=1)

    parser.add_argument(
        "--save_pred_csv",
        type=int,
        default=1,
        help="Save window-level prediction CSV (1=yes, 0=no).",
    )

    parser.add_argument(
        "--save_event_csv",
        type=int,
        default=1,
        help="Save event-level prediction CSV (1=yes, 0=no).",
    )

    return parser.parse_args()


# =========================================================
# REPRODUCIBILITY
# =========================================================
def seed_everything(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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


def make_worker_init_fn(seed: int):
    def _init_fn(worker_id: int):
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _init_fn


def make_torch_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


# =========================================================
# RUN METADATA
# =========================================================
@dataclass
class RunMeta:
    run_id: str
    seed: int
    deterministic: bool
    script: str
    cwd: str
    host: str
    os: str
    python: str
    torch: str
    cuda: str
    cudnn: str
    device: str
    timestamp: str
    config: Dict[str, Any]


def write_run_meta(
    out_dir: Path,
    *,
    seed: int,
    deterministic: bool,
    device: str,
    config: Dict[str, Any],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = RunMeta(
        run_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
        seed=int(seed),
        deterministic=bool(deterministic),
        script=Path(__file__).name,
        cwd=str(Path.cwd()),
        host=platform.node(),
        os=f"{platform.system()} {platform.release()}",
        python=platform.python_version(),
        torch=torch.__version__,
        cuda=str(torch.version.cuda),
        cudnn=str(torch.backends.cudnn.version()),
        device=str(device),
        timestamp=datetime.now().isoformat(timespec="seconds"),
        config=config,
    )

    meta_path = out_dir / f"run_meta_{meta.run_id}_seed{seed}.json"
    meta_path.write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")

    return meta_path


# =========================================================
# METRICS
# =========================================================
def compute_summary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_classes: int = 3,
    malfunction_label: int = 1,
) -> dict:
    """
    Compute summary metrics for the 3-class setting.

    FAR is defined as:
        normal/background samples predicted as malfunction
        divided by
        all normal/background samples.

    Label mapping:
        Class_1 -> 0
        Class_2 -> 1, malfunction
        Class_4 -> 2
    """
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    normal_mask = y_true != malfunction_label
    if normal_mask.sum() > 0:
        far = np.mean(y_pred[normal_mask] == malfunction_label)
    else:
        far = np.nan

    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    roc_auc = roc_auc_score(
        y_true_bin,
        y_prob,
        average="macro",
        multi_class="ovr",
    )

    pr_auc = average_precision_score(
        y_true_bin,
        y_prob,
        average="macro",
    )

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "far": float(far),
        "roc_auc_macro_ovr": float(roc_auc),
        "pr_auc_macro_ovr": float(pr_auc),
    }


def print_summary_metrics(title: str, metrics: dict) -> None:
    print(title)
    print(f"Acc.:    {metrics['accuracy']:.4f}")
    print(f"F1:      {metrics['macro_f1']:.4f}")
    print(f"FAR:     {metrics['far']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc_macro_ovr']:.4f}")
    print(f"PR-AUC:  {metrics['pr_auc_macro_ovr']:.4f}")


# =========================================================
# EVENT-LEVEL TOP-K MEAN AGGREGATION
# =========================================================
def event_level_topk_mean(
    df_meta: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    k: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute event/file-level predictions using Top-K mean aggregation.

    Args:
        df_meta:
            Metadata DataFrame aligned with y_true and y_prob.
            Must contain file_id.
        y_true:
            Window-level true labels.
        y_prob:
            Window-level softmax probabilities with shape (N_windows, N_classes).
        k:
            Number of most confident windows used per file_id.

    Returns:
        y_true_event:
            Event/file-level true labels.
        y_pred_event:
            Event/file-level predicted labels.
        y_prob_event:
            Event/file-level aggregated probability vectors.
    """
    if "file_id" not in df_meta.columns:
        raise ValueError(
            "Metadata CSV must contain a 'file_id' column for event-level aggregation."
        )

    confidence = y_prob.max(axis=1)

    tmp = df_meta.copy()
    tmp["_idx"] = np.arange(len(tmp))
    tmp["_conf"] = confidence
    tmp["_ytrue"] = y_true

    y_true_events = []
    y_pred_events = []
    y_prob_events = []

    for file_id, group in tmp.groupby("file_id", sort=True):
        idxs = group["_idx"].to_numpy()

        # Select Top-K windows by confidence.
        order = np.argsort(group["_conf"].to_numpy())[::-1]
        top = order[: max(1, min(k, len(order)))]
        top_idxs = idxs[top]

        # Top-K mean probability aggregation.
        mean_prob = y_prob[top_idxs].mean(axis=0)
        pred_event = int(mean_prob.argmax())

        # All windows belonging to one file/event should share the same label.
        true_event = int(group["_ytrue"].iloc[0])

        y_true_events.append(true_event)
        y_pred_events.append(pred_event)
        y_prob_events.append(mean_prob)

    return (
        np.asarray(y_true_events, dtype=int),
        np.asarray(y_pred_events, dtype=int),
        np.vstack(y_prob_events),
    )


def build_event_prediction_table(
    df_meta: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    k: int = 5,
) -> pd.DataFrame:
    """
    Build a per-file/event prediction table using Top-K mean aggregation.
    """
    confidence = y_prob.max(axis=1)

    tmp = df_meta.copy()
    tmp["_idx"] = np.arange(len(tmp))
    tmp["_conf"] = confidence

    rows = []

    for file_id, group in tmp.groupby("file_id", sort=True):
        idxs = group["_idx"].to_numpy()

        order = np.argsort(group["_conf"].to_numpy())[::-1]
        top = order[: max(1, min(k, len(order)))]
        top_idxs = idxs[top]

        mean_prob = y_prob[top_idxs].mean(axis=0)

        rows.append(
            {
                "file_id": file_id,
                "n_windows": int(len(group)),
                "k_used": int(len(top_idxs)),
                "y_true_event": int(y_true[top_idxs[0]]),
                "y_pred_event": int(mean_prob.argmax()),
                "p_class1_mean_topk": float(mean_prob[0]),
                "p_class2_mean_topk": float(mean_prob[1]),
                "p_class4_mean_topk": float(mean_prob[2]),
            }
        )

    return pd.DataFrame(rows)


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    args = parse_args()

    test_csv = args.test_csv
    pann_checkpoint = args.pann_checkpoint
    model_weights = args.model_weights
    out_dir = args.out_dir

    topk = int(args.topk)
    batch_size = int(args.batch_size)
    num_workers = int(args.num_workers)
    seed = int(args.seed)
    deterministic = bool(args.deterministic)

    save_pred_csv = bool(args.save_pred_csv)
    save_event_csv = bool(args.save_event_csv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_names = ["Class_1", "Class_2", "Class_4"]

    assert test_csv.exists(), f"Test CSV not found: {test_csv}"
    assert pann_checkpoint.exists(), f"PANN checkpoint not found: {pann_checkpoint}"
    assert model_weights.exists(), f"Model weights not found: {model_weights}"

    out_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # APPLY REPRODUCIBILITY SETTINGS
    # =========================================================
    seed_everything(seed, deterministic=deterministic)
    data_generator = make_torch_generator(seed)
    worker_init_fn = make_worker_init_fn(seed)

    # =========================================================
    # DATA
    # =========================================================
    df_meta = pd.read_csv(test_csv)

    test_ds = WaveformDataset(test_csv, sample_rate=25600)

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,  # critical for alignment with df_meta
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        generator=data_generator,
        worker_init_fn=worker_init_fn,
    )

    print("=== EVALUATION INFO ===")
    print("device:          ", device)
    print("test_csv:        ", test_csv)
    print("pann_checkpoint: ", pann_checkpoint)
    print("model_weights:   ", model_weights)
    print("out_dir:         ", out_dir)
    print("topk:            ", topk)
    print("test samples:    ", len(test_ds))
    print("=======================")

    # =========================================================
    # MODEL
    # =========================================================
    model = CNN14TruncatedFineTune(
        checkpoint_path=pann_checkpoint,
        num_classes=3,
        emb_dim=128,
        freeze_frontend=True,
    )

    state = torch.load(model_weights, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    softmax = torch.nn.Softmax(dim=1)

    # =========================================================
    # WINDOW-LEVEL EVALUATION
    # =========================================================
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for wave, label in test_loader:
            wave = wave.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            logits, _ = model(wave)
            probs = softmax(logits)
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(label.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)

    if len(df_meta) != len(y_true):
        raise RuntimeError(
            f"Metadata rows ({len(df_meta)}) != predictions ({len(y_true)}). "
            "This breaks event-level alignment. Ensure test_csv matches the dataset."
        )

    window_metrics = compute_summary_metrics(
        y_true,
        y_pred,
        y_prob,
        n_classes=3,
        malfunction_label=1,
    )

    window_cm = confusion_matrix(y_true, y_pred)
    window_report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
    )

    print("\n================ WINDOW-LEVEL TEST RESULTS ================")
    print_summary_metrics("Window-level summary metrics:", window_metrics)
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(window_cm)
    print("\nClassification Report:")
    print(window_report)
    print("==========================================================\n")

    # =========================================================
    # EVENT-LEVEL EVALUATION: TOP-K MEAN AGGREGATION
    # =========================================================
    y_true_event, y_pred_event, y_prob_event = event_level_topk_mean(
        df_meta,
        y_true,
        y_prob,
        k=topk,
    )

    event_metrics = compute_summary_metrics(
        y_true_event,
        y_pred_event,
        y_prob_event,
        n_classes=3,
        malfunction_label=1,
    )

    event_cm = confusion_matrix(y_true_event, y_pred_event)
    event_report = classification_report(
        y_true_event,
        y_pred_event,
        target_names=class_names,
        digits=4,
    )

    print("\n=========== EVENT-LEVEL TEST RESULTS (Top-K Mean) ===========")
    print(f"Top-K: {topk}")
    print_summary_metrics("Event-level summary metrics:", event_metrics)
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(event_cm)
    print("\nClassification Report:")
    print(event_report)
    print("============================================================\n")

    # =========================================================
    # SAVE RUN METADATA
    # =========================================================
    meta_path = write_run_meta(
        out_dir,
        seed=seed,
        deterministic=deterministic,
        device=device,
        config={
            "test_csv": str(test_csv),
            "pann_checkpoint": str(pann_checkpoint),
            "model_weights": str(model_weights),
            "batch_size": batch_size,
            "num_workers": num_workers,
            "class_names": class_names,
            "topk": topk,
            "window_metrics": window_metrics,
            "event_metrics_topk_mean": event_metrics,
            "window_confusion_matrix": window_cm.tolist(),
            "event_confusion_matrix": event_cm.tolist(),
        },
    )
    print(f"[RUN] metadata saved: {meta_path}")

    # =========================================================
    # SAVE WINDOW-LEVEL PREDICTIONS
    # =========================================================
    if save_pred_csv:
        df_pred = df_meta.copy()
        df_pred["y_true"] = y_true.astype(int)
        df_pred["y_pred"] = y_pred.astype(int)
        df_pred["p_class1"] = y_prob[:, 0]
        df_pred["p_class2"] = y_prob[:, 1]
        df_pred["p_class4"] = y_prob[:, 2]

        out_pred = out_dir / f"test_predictions_window_seed{seed}.csv"
        df_pred.to_csv(out_pred, index=False)

        print(f"[SAVE] window predictions saved: {out_pred}")

    # =========================================================
    # SAVE EVENT-LEVEL PREDICTIONS
    # =========================================================
    if save_event_csv:
        df_event = build_event_prediction_table(
            df_meta,
            y_true,
            y_prob,
            k=topk,
        )

        out_event = out_dir / f"test_predictions_event_topk{topk}_mean_seed{seed}.csv"
        df_event.to_csv(out_event, index=False)

        print(f"[SAVE] event predictions saved: {out_event}")

    print("[DONE]")


if __name__ == "__main__":
    main()