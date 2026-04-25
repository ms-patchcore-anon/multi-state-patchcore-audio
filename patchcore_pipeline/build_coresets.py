"""
build_coresets.py
--------------------

Build per-class PatchCore memory banks using k-center greedy coreset selection.

This script builds three separate class-specific memory banks:
- Class_1
- Class_2
- Class_4

The input is the train index CSV produced by:

    patchcore_pipeline/extract_embeddings.py

For each class, the script:
1. Loads all train embeddings.
2. Saves the full memory bank.
3. Selects a coreset using k-center greedy.
4. Saves the coreset bank and selected indices.
5. Saves metadata rows for traceability.

Input
-----
- <patchcore_output_dir>/indices/train_index.csv
- <patchcore_output_dir>/embeddings_full/train/Class_X/emb_*.npy

Output
------
- <patchcore_output_dir>/coresets/Class_X_full.npy
- <patchcore_output_dir>/coresets/Class_X_coreset.npy
- <patchcore_output_dir>/coresets/Class_X_coreset_idx.npy
- <patchcore_output_dir>/coresets/Class_X_coreset_meta.csv

Example usage for 0.2 s windows:
    python patchcore_pipeline/02_build_coresets.py \
        --input_dir outputs/patchcore_0p2s \
        --k_coreset 1000

Example usage for 0.4 s windows:
    python patchcore_pipeline/02_build_coresets.py \
        --input_dir outputs/patchcore_0p4s \
        --k_coreset 1000
"""

from __future__ import annotations

from pathlib import Path
import argparse
import os
import random

import numpy as np
import pandas as pd
from tqdm import tqdm


# =========================================================
# ARGUMENTS
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build per-class PatchCore coreset memory banks."
    )

    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("outputs/patchcore_0p2s"),
        help=(
            "PatchCore output directory produced by 01_extract_embeddings.py. "
            "Must contain indices/train_index.csv and embeddings_full/."
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help=(
            "Directory where coreset files will be saved. "
            "If omitted, defaults to <input_dir>/coresets."
        ),
    )

    parser.add_argument("--k_coreset", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# =========================================================
# REPRODUCIBILITY
# =========================================================
def seed_everything(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


# =========================================================
# HELPERS
# =========================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_npy(path: Path) -> np.ndarray:
    return np.load(path)


def save_npy(path: Path, array: np.ndarray) -> None:
    ensure_dir(path.parent)
    np.save(path, array)


def kcenter_greedy(
    x: np.ndarray,
    *,
    k: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Select k representative samples using greedy k-center selection.

    Args:
        x:
            Embedding matrix of shape (N, D).
        k:
            Number of samples to select.
        seed:
            Random seed used for the first selected point.

    Returns:
        selected_indices:
            Array of selected indices into x.
    """
    n = x.shape[0]

    if k >= n:
        return np.arange(n, dtype=np.int64)

    rng = np.random.default_rng(seed)

    selected = []
    first_idx = int(rng.integers(0, n))
    selected.append(first_idx)

    # Squared Euclidean distance to the nearest selected center.
    diff = x - x[first_idx]
    min_dist = np.sum(diff * diff, axis=1)

    for _ in tqdm(range(1, k), desc="k-center greedy"):
        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)

        diff = x - x[next_idx]
        dist = np.sum(diff * diff, axis=1)
        min_dist = np.minimum(min_dist, dist)

    return np.asarray(selected, dtype=np.int64)


def resolve_embedding_path(row: pd.Series) -> Path:
    """
    Resolve an embedding file path from a train index row.

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


def load_class_embeddings_from_index(df_cls: pd.DataFrame) -> np.ndarray:
    """
    Load embeddings in the exact order they appear in the index CSV.
    This makes the memory bank reproducible and traceable.
    """
    class_name = str(df_cls["class_name"].iloc[0])

    embeddings = []

    for _, row in tqdm(
        df_cls.iterrows(),
        total=len(df_cls),
        desc=f"Loading {class_name}",
    ):
        emb_path = resolve_embedding_path(row)

        if not emb_path.exists():
            raise FileNotFoundError(f"Missing embedding file: {emb_path}")

        embeddings.append(load_npy(emb_path))

    return np.stack(embeddings, axis=0)


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    args = parse_args()

    seed_everything(args.seed)

    input_dir = args.input_dir
    index_csv = input_dir / "indices" / "train_index.csv"

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = input_dir / "coresets"

    ensure_dir(output_dir)

    k_coreset = int(args.k_coreset)
    seed = int(args.seed)

    classes = ["Class_1", "Class_2", "Class_4"]

    if not index_csv.exists():
        raise FileNotFoundError(f"Missing train index CSV: {index_csv}")

    df = pd.read_csv(index_csv)

    required_cols = {"class_name"}
    missing = required_cols - set(df.columns)
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

    if "split" in df.columns:
        unique_splits = sorted(df["split"].dropna().unique().tolist())
        if unique_splits != ["train"]:
            raise ValueError(
                f"Expected train-only index, but found split values: {unique_splits}"
            )

    print("=== CORESET BUILD INFO ===")
    print("input_dir:   ", input_dir)
    print("index_csv:   ", index_csv)
    print("output_dir:  ", output_dir)
    print("k_coreset:   ", k_coreset)
    print("seed:        ", seed)
    print("==========================")

    for class_name in classes:
        df_cls = df[df["class_name"] == class_name].reset_index(drop=True)

        if len(df_cls) == 0:
            raise RuntimeError(f"No rows for {class_name} found in {index_csv}")

        embeddings = load_class_embeddings_from_index(df_cls)
        print(f"\n[INFO] {class_name}: full bank shape = {embeddings.shape}")

        full_bank_path = output_dir / f"{class_name}_full.npy"
        save_npy(full_bank_path, embeddings)

        k = min(k_coreset, embeddings.shape[0])

        selected_idx = kcenter_greedy(
            embeddings,
            k=k,
            seed=seed,
        )

        coreset = embeddings[selected_idx]

        coreset_path = output_dir / f"{class_name}_coreset.npy"
        idx_path = output_dir / f"{class_name}_coreset_idx.npy"
        meta_path = output_dir / f"{class_name}_coreset_meta.csv"

        save_npy(coreset_path, coreset)
        save_npy(idx_path, selected_idx)

        df_coreset_meta = df_cls.iloc[selected_idx].copy()
        df_coreset_meta.to_csv(meta_path, index=False)

        print(f"[OK] {class_name}: coreset shape = {coreset.shape} (k={k})")
        print(f"[OK] Saved full bank: {full_bank_path}")
        print(f"[OK] Saved coreset:   {coreset_path}")
        print(f"[OK] Saved indices:   {idx_path}")
        print(f"[OK] Saved metadata:  {meta_path}")

    print("\n[DONE] Coreset memory banks built.")


if __name__ == "__main__":
    main()