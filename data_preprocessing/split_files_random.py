"""
split_files_random.py
---------------------

Create a reproducible file-level train/validation/test split for extracted
waveform windows.

This script is shared by both the 0.2 s and 0.4 s preprocessing pipelines.
Use --dataset_root to select the corresponding extracted-window dataset.

Default input path points to the 0.2 s sample output folder:
    outputs/sample_windows_0p2s_3class/

For 0.4 s windows, pass:
    --dataset_root outputs/sample_windows_0p4s_3class

For the full private dataset, pass the extracted-window dataset root via:
    --dataset_root

Expected input structure:
- <dataset_root>/
    Class_1/  -- noisy background
    Class_2/  -- malfunction events
    Class_4/  -- clean background

Output:
- <dataset_root>/File_split_CSVs/file_split.csv
- <dataset_root>/File_split_CSVs/file_split_runinfo.json

Example usage for 0.2 s windows:
    python data_preprocessing/split_files_random.py \
        --dataset_root outputs/sample_windows_0p2s_3class

Example usage for 0.4 s windows:
    python data_preprocessing/split_files_random.py \
        --dataset_root outputs/sample_windows_0p4s_3class
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# ARGUMENTS
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="outputs/sample_windows_0p2s_3class",
        help="Dataset root containing Class_1, Class_2, and Class_4 folders.",
    )
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        type=int,
        default=1,
        help="Overwrite output files if they already exist (1=yes, 0=no).",
    )
    return parser.parse_args()


# =========================================================
# HELPERS
# =========================================================
def safe_extract_file_id(wav_path: Path) -> int | None:
    """
    Extract file_id from a filename of the form:
    Class_X_<file_id>_<start>_<idx>.wav
    """
    parts = wav_path.stem.split("_")
    if len(parts) < 3:
        return None

    try:
        return int(parts[2])
    except ValueError:
        return None


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    csv_dir = dataset_root / "File_split_CSVs"
    csv_dir.mkdir(parents=True, exist_ok=True)

    out_csv = csv_dir / "file_split.csv"
    runinfo_path = csv_dir / "file_split_runinfo.json"

    train_ratio = float(args.train_ratio)
    val_ratio = float(args.val_ratio)
    seed = int(args.seed)
    overwrite = bool(args.overwrite)

    if train_ratio <= 0 or val_ratio < 0:
        raise ValueError("train_ratio must be > 0 and val_ratio must be >= 0.")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0.")

    if out_csv.exists() and not overwrite:
        print(f"[SKIP] {out_csv} already exists and overwrite is disabled.")
        return

    file_ids: set[int] = set()
    skipped = 0

    for cls in ["Class_1", "Class_2", "Class_4"]:
        cls_dir = dataset_root / cls
        if not cls_dir.exists():
            raise FileNotFoundError(f"Missing folder: {cls_dir}")

        for wav in cls_dir.glob("*.wav"):
            fid = safe_extract_file_id(wav)
            if fid is None:
                skipped += 1
                continue
            file_ids.add(fid)

    file_ids = np.array(sorted(file_ids), dtype=int)

    if len(file_ids) == 0:
        raise RuntimeError("No valid file_ids found.")

    print(f"[INFO] Found {len(file_ids)} unique file_ids")
    if skipped:
        print(f"[WARN] Skipped {skipped} files with unexpected names")

    rng = np.random.default_rng(seed)
    rng.shuffle(file_ids)

    n_total = len(file_ids)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)

    if n_total >= 3:
        n_train = max(n_train, 1)
        n_val = max(n_val, 1)
        if n_train + n_val >= n_total:
            n_val = max(1, n_total - n_train - 1)

    train_ids = file_ids[:n_train]
    val_ids = file_ids[n_train:n_train + n_val]
    test_ids = file_ids[n_train + n_val:]

    rows = []
    rows += [{"file_id": int(fid), "split": "train"} for fid in train_ids]
    rows += [{"file_id": int(fid), "split": "val"} for fid in val_ids]
    rows += [{"file_id": int(fid), "split": "test"} for fid in test_ids]

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    runinfo = {
        "timestamp": datetime.now().isoformat(),
        "dataset_root": str(dataset_root),
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": 1.0 - train_ratio - val_ratio,
        "n_total_file_ids": int(n_total),
        "n_train": int(len(train_ids)),
        "n_val": int(len(val_ids)),
        "n_test": int(len(test_ids)),
        "out_csv": str(out_csv),
        "skipped_bad_filenames": int(skipped),
    }
    runinfo_path.write_text(json.dumps(runinfo, indent=2), encoding="utf-8")

    print(f"[OK] Split saved to: {out_csv}")
    print(f"[OK] Run info saved to: {runinfo_path}")
    print(df["split"].value_counts())


if __name__ == "__main__":
    main()