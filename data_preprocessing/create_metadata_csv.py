"""
create_metadata_csv.py
----------------------

Create a metadata CSV for extracted log-mel files.

The output CSV contains the following columns:
- path
- label
- class_name
- file_id
- split

Input:
- outputs/sample_windows_0p2s_3class/File_split_CSVs/file_split.csv
- outputs/sample_windows_0p2s_3class/Class_1_log-mels/
- outputs/sample_windows_0p2s_3class/Class_2_log-mels/
- outputs/sample_windows_0p2s_3class/Class_4_log-mels/

Output:
- outputs/sample_windows_0p2s_3class/File_split_CSVs/metadata_logmel.csv
"""

from pathlib import Path
import argparse

import pandas as pd


# =========================================================
# PATHS
# =========================================================
# Default path points to the sample output folder created by the data preparation scripts.
# For the full private dataset, provide --dataset_root from the command line.

parser = argparse.ArgumentParser(
    description="Create a metadata CSV for extracted log-mel files."
)

parser.add_argument(
    "--dataset_root",
    type=Path,
    default=Path("outputs/sample_windows_0p2s_3class"),
    help="Dataset root containing log-mel folders and File_split_CSVs.",
)

args = parser.parse_args()

DATASET = args.dataset_root
CSV_DIR = DATASET / "File_split_CSVs"

SPLIT_CSV = CSV_DIR / "file_split.csv"
OUT_CSV = CSV_DIR / "metadata_logmel.csv"


# =========================================================
# CLASS LABELS
# =========================================================
CLASS_INFO = {
    "Class_1": 0,
    "Class_2": 1,
    "Class_4": 2,
}


# =========================================================
# LOAD FILE SPLIT
# =========================================================
split_df = pd.read_csv(SPLIT_CSV)
fileid_to_split = dict(zip(split_df["file_id"], split_df["split"]))


# =========================================================
# HELPER
# =========================================================
def extract_file_id_from_name(stem: str) -> int:
    """
    Extract the file_id from a filename stem of the form:
    Class_2_100_46947_0238
    """
    parts = stem.split("_")
    return int(parts[2])


# =========================================================
# BUILD METADATA
# =========================================================
rows = []

for class_name, label in CLASS_INFO.items():
    logmel_dir = DATASET / f"{class_name}_log-mels"

    for npy_path in sorted(logmel_dir.glob("*.npy")):
        file_id = extract_file_id_from_name(npy_path.stem)

        if file_id not in fileid_to_split:
            raise ValueError(
                f"file_id={file_id} not found in split CSV."
            )

        split = fileid_to_split[file_id]

        rows.append(
            {
                "path": str(npy_path),
                "label": label,
                "class_name": class_name,
                "file_id": file_id,
                "split": split,
            }
        )

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)

print(f"Metadata CSV saved to: {OUT_CSV}")
print("\nSamples per split:")
print(df["split"].value_counts())
print("\nSamples per class:")
print(df["class_name"].value_counts())