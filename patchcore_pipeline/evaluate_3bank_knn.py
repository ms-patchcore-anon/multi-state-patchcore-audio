"""
evaluate_3bank_knn.py
------------------------

Evaluate a 3-bank PatchCore-style nearest-neighbor pipeline.

This script reports both:

1) Window-level metrics
   - Accuracy
   - Macro-F1
   - FAR
   - Macro ROC-AUC
   - Macro PR-AUC
   - Confusion matrix
   - Classification report

2) Event/file-level metrics using Top-5 mean aggregation grouped by file_id
   - Accuracy
   - Macro-F1
   - FAR
   - Macro ROC-AUC
   - Macro PR-AUC
   - Confusion matrix
   - Classification report

PatchCore scoring:
- Each window embedding is compared against three class-specific memory banks:
    Class_1
    Class_2
    Class_4

- For each class bank, the window distance is computed as the mean of the
  top-k nearest neighbor distances inside that bank.

- Window prediction:
    class with minimum distance

- Event-level Top-5 mean aggregation:
    For each file_id and each class, select the Top-5 smallest window distances
    and average them. The event prediction is the class with the minimum
    aggregated distance.

FAR definition:
- Class_2 is treated as the malfunction class.
- Class_1 and Class_4 are treated as normal/background classes.
- FAR = normal/background samples predicted as malfunction / all normal/background samples.

Expected inputs
---------------
Produced by:
    patchcore_pipeline/01_extract_embeddings.py
    patchcore_pipeline/02_build_coresets.py

Input structure:
- <input_dir>/indices/test_index.csv
- <input_dir>/coresets/Class_1_coreset.npy
- <input_dir>/coresets/Class_2_coreset.npy
- <input_dir>/coresets/Class_4_coreset.npy
- embeddings referenced by test_index.csv

Example usage for 0.2 s windows:
    python patchcore_pipeline/03_evaluate_3bank_knn.py \
        --input_dir outputs/patchcore_0p2s \
        --split test \
        --topk_neighbors 5 \
        --event_topk 5

Example usage for 0.4 s windows:
    python patchcore_pipeline/03_evaluate_3bank_knn.py \
        --input_dir outputs/patchcore_0p4s \
        --split test \
        --topk_neighbors 5 \
        --event_topk 5
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize


# =========================================================
# ARGUMENTS
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate 3-bank PatchCore KNN at window and event level."
    )

    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("outputs/patchcore_0p2s"),
        help=(
            "PatchCore output directory containing indices/, coresets/, "
            "and embeddings_full/."
        ),
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Split to evaluate.",
    )

    parser.add_argument(
        "--topk_neighbors",
        type=int,
        default=5,
        help="Number of nearest neighbors inside each class bank for window scoring.",
    )

    parser.add_argument(
        "--event_topk",
        type=int,
        default=5,
        help="Number of best windows per event/class used for Top-K mean aggregation.",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory for evaluation outputs. Defaults to <input_dir>/results.",
    )

    return parser.parse_args()


# =========================================================
# CONSTANTS
# =========================================================
CLASS_NAMES = ["Class_1", "Class_2", "Class_4"]
CLASS_TO_LABEL = {"Class_1": 0, "Class_2": 1, "Class_4": 2}
MALFUNCTION_LABEL = 1


# =========================================================
# HELPERS
# =========================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_npy(path: Path) -> np.ndarray:
    return np.load(path)


def save_json(path: Path, data: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def resolve_embedding_path(row: pd.Series) -> Path:
    """
    Resolve an embedding path from an index row.

    Preferred:
        embedding_dir + embedding_file

    Fallback:
        embedding_relpath
    """
    if "embedding_dir" in row and "embedding_file" in row:
        return Path(str(row["embedding_dir"])) / str(row["embedding_file"])

    if "embedding_relpath" in row:
        return Path(str(row["embedding_relpath"]))

    raise ValueError(
        "Index row must contain either embedding_dir + embedding_file "
        "or embedding_relpath."
    )


def topk_mean_distance_to_bank(
    z: np.ndarray,
    bank: np.ndarray,
    *,
    k: int = 5,
) -> float:
    """
    Compute mean distance to the k nearest neighbors in one memory bank.

    Args:
        z:
            Embedding vector of shape (D,).
        bank:
            Memory bank of shape (N, D).
        k:
            Number of nearest neighbors.

    Returns:
        Mean of k smallest Euclidean distances.
    """
    diff = bank - z[None, :]
    distances = np.sqrt(np.sum(diff * diff, axis=1))

    k_eff = min(max(1, k), distances.shape[0])
    nearest = np.partition(distances, k_eff - 1)[:k_eff]

    return float(np.mean(nearest))


def score_embedding_3bank(
    z: np.ndarray,
    banks: dict[str, np.ndarray],
    *,
    topk_neighbors: int = 5,
) -> dict[str, float]:
    """
    Return distance to each class-specific memory bank.
    Lower distance means more similar.
    """
    return {
        class_name: topk_mean_distance_to_bank(
            z,
            bank,
            k=topk_neighbors,
        )
        for class_name, bank in banks.items()
    }


def distances_to_scores(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Convert distances to scores for ROC-AUC and PR-AUC.

    Higher score should mean stronger evidence for the class.
    Therefore, score = -distance.
    """
    return -distance_matrix


def compute_summary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    *,
    n_classes: int = 3,
    malfunction_label: int = 1,
) -> dict:
    """
    Compute table-ready metrics:
    - Accuracy
    - Macro-F1
    - FAR
    - Macro ROC-AUC
    - Macro PR-AUC
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
        y_score,
        average="macro",
        multi_class="ovr",
    )

    pr_auc = average_precision_score(
        y_true_bin,
        y_score,
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


def extract_file_id_from_path(path_value: str) -> str:
    """
    Fallback file_id extraction from filenames like:
        Class_1_115_67438_0016.wav
    """
    stem = Path(str(path_value)).stem
    parts = stem.split("_")

    if len(parts) >= 3 and parts[0] == "Class":
        return str(parts[2])

    return stem


def validate_index(df: pd.DataFrame, index_csv: Path) -> pd.DataFrame:
    """
    Validate and standardize the evaluation index.
    """
    required = {"class_name", "path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{index_csv} missing required columns: {missing}")

    if not (
        {"embedding_dir", "embedding_file"}.issubset(df.columns)
        or "embedding_relpath" in df.columns
    ):
        raise ValueError(
            f"{index_csv} must contain either "
            "embedding_dir + embedding_file or embedding_relpath."
        )

    df = df[df["class_name"].isin(CLASS_NAMES)].reset_index(drop=True).copy()

    if "file_id" not in df.columns:
        df["file_id"] = df["path"].map(extract_file_id_from_path)

    df["file_id"] = df["file_id"].astype(str)

    return df


# =========================================================
# WINDOW-LEVEL SCORING
# =========================================================
def evaluate_windows(
    df_idx: pd.DataFrame,
    banks: dict[str, np.ndarray],
    *,
    topk_neighbors: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Score every window against all class memory banks.

    Returns:
        window_df:
            Per-window prediction table.
        y_true:
            Integer true labels.
        y_pred:
            Integer predicted labels.
        distance_matrix:
            Shape (N, 3), lower is better.
        score_matrix:
            Shape (N, 3), higher is better.
    """
    rows = []
    y_true = []
    y_pred = []
    distances_all = []

    for _, row in tqdm(
        df_idx.iterrows(),
        total=len(df_idx),
        desc="Window-level PatchCore scoring",
    ):
        emb_path = resolve_embedding_path(row)

        if not emb_path.exists():
            raise FileNotFoundError(f"Missing embedding file: {emb_path}")

        z = load_npy(emb_path)

        distance_dict = score_embedding_3bank(
            z,
            banks,
            topk_neighbors=topk_neighbors,
        )

        distances = np.asarray(
            [distance_dict[c] for c in CLASS_NAMES],
            dtype=float,
        )

        pred_label = int(np.argmin(distances))
        true_label = CLASS_TO_LABEL[row["class_name"]]

        y_true.append(true_label)
        y_pred.append(pred_label)
        distances_all.append(distances)

        rows.append(
            {
                "split": row["split"] if "split" in row else "",
                "path": row["path"],
                "class_name": row["class_name"],
                "file_id": row["file_id"],
                "y_true": true_label,
                "y_pred": pred_label,
                "pred_class": CLASS_NAMES[pred_label],
                "dist_Class_1": float(distances[0]),
                "dist_Class_2": float(distances[1]),
                "dist_Class_4": float(distances[2]),
                "score_Class_1": float(-distances[0]),
                "score_Class_2": float(-distances[1]),
                "score_Class_4": float(-distances[2]),
            }
        )

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    distance_matrix = np.vstack(distances_all)
    score_matrix = distances_to_scores(distance_matrix)

    window_df = pd.DataFrame(rows)

    return window_df, y_true, y_pred, distance_matrix, score_matrix


# =========================================================
# EVENT-LEVEL TOP-5 MEAN AGGREGATION
# =========================================================
def event_level_topk_mean(
    window_df: pd.DataFrame,
    *,
    event_topk: int = 5,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate window distances into event/file-level predictions.

    For each file_id and each class:
    - take the Top-K smallest window distances
    - average them
    - predict the class with minimum aggregated distance

    Returns:
        event_df:
            Per-event prediction table.
        y_true_event:
            Integer true labels.
        y_pred_event:
            Integer predicted labels.
        event_distances:
            Shape (N_events, 3), lower is better.
        event_scores:
            Shape (N_events, 3), higher is better.
    """
    event_rows = []
    y_true_event = []
    y_pred_event = []
    event_distances_all = []

    for file_id, group in tqdm(
        window_df.groupby("file_id", sort=True),
        desc="Event-level Top-K mean aggregation",
    ):
        true_labels = sorted(group["y_true"].unique().tolist())

        if len(true_labels) != 1:
            # This should not happen if file_id is a clean event/file key.
            # Skipping avoids corrupt event-level metrics.
            print(
                f"[WARN] Skipping file_id={file_id}: "
                f"multiple true labels found: {true_labels}"
            )
            continue

        true_label = int(true_labels[0])

        aggregated_distances = []

        for class_name in CLASS_NAMES:
            dist_col = f"dist_{class_name}"
            dists = group[dist_col].to_numpy(dtype=float)

            k_eff = min(max(1, event_topk), len(dists))
            topk_smallest = np.partition(dists, k_eff - 1)[:k_eff]
            aggregated_distances.append(float(np.mean(topk_smallest)))

        aggregated_distances = np.asarray(aggregated_distances, dtype=float)
        pred_label = int(np.argmin(aggregated_distances))

        y_true_event.append(true_label)
        y_pred_event.append(pred_label)
        event_distances_all.append(aggregated_distances)

        event_rows.append(
            {
                "file_id": file_id,
                "n_windows": int(len(group)),
                "k_used": int(min(max(1, event_topk), len(group))),
                "y_true_event": true_label,
                "y_pred_event": pred_label,
                "true_class": CLASS_NAMES[true_label],
                "pred_class": CLASS_NAMES[pred_label],
                "topk_mean_dist_Class_1": float(aggregated_distances[0]),
                "topk_mean_dist_Class_2": float(aggregated_distances[1]),
                "topk_mean_dist_Class_4": float(aggregated_distances[2]),
                "score_Class_1": float(-aggregated_distances[0]),
                "score_Class_2": float(-aggregated_distances[1]),
                "score_Class_4": float(-aggregated_distances[2]),
            }
        )

    y_true_event = np.asarray(y_true_event, dtype=int)
    y_pred_event = np.asarray(y_pred_event, dtype=int)
    event_distances = np.vstack(event_distances_all)
    event_scores = distances_to_scores(event_distances)

    event_df = pd.DataFrame(event_rows)

    return event_df, y_true_event, y_pred_event, event_distances, event_scores


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    args = parse_args()

    input_dir = args.input_dir
    split = args.split
    topk_neighbors = int(args.topk_neighbors)
    event_topk = int(args.event_topk)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = input_dir / "results"

    ensure_dir(output_dir)

    index_csv = input_dir / "indices" / f"{split}_index.csv"
    coreset_dir = input_dir / "coresets"

    if not index_csv.exists():
        raise FileNotFoundError(f"Missing index CSV: {index_csv}")

    for class_name in CLASS_NAMES:
        bank_path = coreset_dir / f"{class_name}_coreset.npy"
        if not bank_path.exists():
            raise FileNotFoundError(f"Missing coreset bank: {bank_path}")

    print("=== PATCHCORE EVALUATION INFO ===")
    print("input_dir:      ", input_dir)
    print("split:          ", split)
    print("index_csv:      ", index_csv)
    print("coreset_dir:    ", coreset_dir)
    print("output_dir:     ", output_dir)
    print("topk_neighbors: ", topk_neighbors)
    print("event_topk:     ", event_topk)
    print("=================================")

    # ---------------------------------------------------------
    # Load memory banks
    # ---------------------------------------------------------
    banks = {
        class_name: load_npy(coreset_dir / f"{class_name}_coreset.npy")
        for class_name in CLASS_NAMES
    }

    print("[INFO] Memory bank shapes:")
    for class_name, bank in banks.items():
        print(f"  {class_name}: {bank.shape}")

    # ---------------------------------------------------------
    # Load index
    # ---------------------------------------------------------
    df_idx = pd.read_csv(index_csv)
    df_idx = validate_index(df_idx, index_csv)

    print(f"[INFO] Number of windows: {len(df_idx)}")
    print(f"[INFO] Number of file_id groups: {df_idx['file_id'].nunique()}")

    # ---------------------------------------------------------
    # Window-level evaluation
    # ---------------------------------------------------------
    (
        window_df,
        y_true_window,
        y_pred_window,
        window_distances,
        window_scores,
    ) = evaluate_windows(
        df_idx,
        banks,
        topk_neighbors=topk_neighbors,
    )

    window_metrics = compute_summary_metrics(
        y_true_window,
        y_pred_window,
        window_scores,
        n_classes=3,
        malfunction_label=MALFUNCTION_LABEL,
    )

    window_cm = confusion_matrix(y_true_window, y_pred_window)
    window_report = classification_report(
        y_true_window,
        y_pred_window,
        target_names=CLASS_NAMES,
        digits=4,
    )

    print("\n================ WINDOW-LEVEL PATCHCORE RESULTS ================")
    print_summary_metrics("Window-level summary metrics:", window_metrics)
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(window_cm)
    print("\nClassification Report:")
    print(window_report)
    print("================================================================\n")

    window_csv = output_dir / f"{split}_window_predictions_topk{topk_neighbors}.csv"
    window_df.to_csv(window_csv, index=False)
    print(f"[SAVE] Window-level predictions saved to: {window_csv}")

    # ---------------------------------------------------------
    # Event-level evaluation with Top-5 mean aggregation
    # ---------------------------------------------------------
    (
        event_df,
        y_true_event,
        y_pred_event,
        event_distances,
        event_scores,
    ) = event_level_topk_mean(
        window_df,
        event_topk=event_topk,
    )

    event_metrics = compute_summary_metrics(
        y_true_event,
        y_pred_event,
        event_scores,
        n_classes=3,
        malfunction_label=MALFUNCTION_LABEL,
    )

    event_cm = confusion_matrix(y_true_event, y_pred_event)
    event_report = classification_report(
        y_true_event,
        y_pred_event,
        target_names=CLASS_NAMES,
        digits=4,
    )

    print("\n============= EVENT-LEVEL PATCHCORE RESULTS (Top-K Mean) =============")
    print(f"Event Top-K: {event_topk}")
    print_summary_metrics("Event-level summary metrics:", event_metrics)
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(event_cm)
    print("\nClassification Report:")
    print(event_report)
    print("======================================================================\n")

    event_csv = (
        output_dir
        / f"{split}_event_predictions_topk{topk_neighbors}_eventTop{event_topk}_mean.csv"
    )
    event_df.to_csv(event_csv, index=False)
    print(f"[SAVE] Event-level predictions saved to: {event_csv}")

    # ---------------------------------------------------------
    # Save metrics JSON
    # ---------------------------------------------------------
    metrics_json = output_dir / f"{split}_metrics_window_and_event.json"

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "split": split,
        "class_names": CLASS_NAMES,
        "malfunction_label": MALFUNCTION_LABEL,
        "settings": {
            "topk_neighbors": topk_neighbors,
            "event_topk": event_topk,
            "aggregation": "Top-K mean of smallest distances per class grouped by file_id",
        },
        "window_level": {
            "metrics": window_metrics,
            "confusion_matrix": window_cm.tolist(),
            "prediction_csv": str(window_csv),
        },
        "event_level": {
            "metrics": event_metrics,
            "confusion_matrix": event_cm.tolist(),
            "prediction_csv": str(event_csv),
        },
        "paths": {
            "input_dir": str(input_dir),
            "index_csv": str(index_csv),
            "coreset_dir": str(coreset_dir),
            "output_dir": str(output_dir),
        },
    }

    save_json(metrics_json, summary)
    print(f"[SAVE] Metrics JSON saved to: {metrics_json}")

    print("[DONE]")


if __name__ == "__main__":
    main()