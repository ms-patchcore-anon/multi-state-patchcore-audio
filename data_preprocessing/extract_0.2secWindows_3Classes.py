"""
extract_0.2secWindows_3Classes.py
---------------------------------

Generate fixed-length 0.2 s waveform windows from annotated industrial audio.

Input:
- WAV files
- Annotation CSV

Output:
Extracted 0.2 s waveform windows saved into class-specific folders:
    Class_1/  -- noisy background
    Class_2/  -- malfunction events
    Class_4/  -- clean background

Classes:
- Class_1: noisy background
- Class_2: malfunction events
- Class_4: clean background
"""

from pathlib import Path
import argparse

import librosa
import numpy as np
import pandas as pd
import soundfile as sf


# =========================================================
# PATHS
# =========================================================
# These default paths are for the anonymized sample dataset in the repository.
# For the full private dataset, provide paths through command-line arguments.

parser = argparse.ArgumentParser(
    description="Generate fixed-length 0.2 s waveform windows from annotated audio."
)

parser.add_argument(
    "--wav_dir",
    type=Path,
    default=Path("AudioDataset/Sample Audios"),
    help="Directory containing input WAV files.",
)

parser.add_argument(
    "--csv_path",
    type=Path,
    default=Path("AudioDataset/Sample Annotations/matching_df_one_event_per_row_3_subset.csv"),
    help="Path to annotation CSV file.",
)

parser.add_argument(
    "--out_base",
    type=Path,
    default=Path("outputs/sample_windows_0p2s_3class"),
    help="Output directory for extracted 0.2 s windows.",
)

args = parser.parse_args()

WAV_DIR = args.wav_dir
CSV_PATH = args.csv_path
OUT_BASE = args.out_base

CLASS_FOLDERS = {
    "Class_1": OUT_BASE / "Class_1",
    "Class_2": OUT_BASE / "Class_2",
    "Class_4": OUT_BASE / "Class_4",
}


# =========================================================
# CONFIG
# =========================================================
SR = 25600
WIN_SEC = 0.2
WIN_SAMP = int(WIN_SEC * SR)

HOP_SEC = 0.1
HOP_SAMP = int(HOP_SEC * SR)

EXCLUDE_MARGIN = 0.05
K_JITTER = 3


# =========================================================
# HELPERS
# =========================================================
def find_wav_file(file_number: int) -> Path | None:
    """Return the matching WAV file for a given file id."""
    f1 = WAV_DIR / f"{int(file_number)}_S1.wav"
    if f1.exists():
        return f1

    f2 = WAV_DIR / f"{int(file_number)}_S2.wav"
    if f2.exists():
        return f2

    matches = sorted(WAV_DIR.glob(f"{int(file_number)}_S*.wav"))
    return matches[0] if matches else None


def clamp_centered_window(center_sec: float, signal_len: int) -> tuple[int, int]:
    """Return a valid centered window [start, end) in samples."""
    center_sample = int(center_sec * SR)
    start = center_sample - WIN_SAMP // 2
    start = max(0, min(start, signal_len - WIN_SAMP))
    end = start + WIN_SAMP
    return start, end


def overlaps(interval: tuple[float, float], forbidden_intervals: list[tuple[float, float]]) -> bool:
    """Return True if interval overlaps any forbidden interval."""
    a, b = interval
    for s, e in forbidden_intervals:
        if not (b <= s or a >= e):
            return True
    return False


# =========================================================
# LOAD ANNOTATIONS
# =========================================================
df = pd.read_csv(CSV_PATH)

windows = {
    "Class_1": [],
    "Class_2": [],
    "Class_4": [],
}


# =========================================================
# EXTRACT WINDOWS
# =========================================================
for file_id, group in df.groupby("File_name"):
    wav_path = find_wav_file(file_id)
    if wav_path is None:
        print(f"[WARN] Missing WAV for File_name={file_id}")
        continue

    y, _ = librosa.load(wav_path, sr=SR, mono=True)
    y_len = len(y)

    class2_rows = group[group["Class"] == "Class_2"]

    exclusion_zones = []
    for _, row in class2_rows.iterrows():
        start = float(row["Time_From_Reference"]) - EXCLUDE_MARGIN
        end = float(row["Time_To_Reference"]) + EXCLUDE_MARGIN
        exclusion_zones.append((start, end))

    # Class_2: centered windows with small temporal offsets
    for _, row in class2_rows.iterrows():
        center = (
            float(row["Time_From_Reference"]) + float(row["Time_To_Reference"])
        ) / 2.0

        max_jitter = min(0.05, WIN_SEC / 4)
        offsets = np.linspace(-max_jitter, max_jitter, K_JITTER)

        for offset in offsets:
            st, en = clamp_centered_window(center + float(offset), y_len)
            seg = y[st:en]
            if len(seg) == WIN_SAMP:
                windows["Class_2"].append((file_id, st, seg))

    # Class_1: sliding windows within annotated ranges
    for _, row in group[group["Class"] == "Class_1"].iterrows():
        start_sec = float(row["Time_From_Reference"])
        end_sec = float(row["Time_To_Reference"])

        start_samp = int(start_sec * SR)
        end_samp = int(end_sec * SR)

        for st in range(start_samp, end_samp - WIN_SAMP + 1, HOP_SAMP):
            en = st + WIN_SAMP
            if en > y_len:
                continue
            if overlaps((st / SR, en / SR), exclusion_zones):
                continue

            seg = y[st:en]
            if len(seg) == WIN_SAMP:
                windows["Class_1"].append((file_id, st, seg))

    # Class_4: sliding windows within annotated ranges
    for _, row in group[group["Class"] == "Class_4"].iterrows():
        start_sec = float(row["Time_From_Reference"])
        end_sec = float(row["Time_To_Reference"])

        start_samp = int(start_sec * SR)
        end_samp = int(end_sec * SR)

        for st in range(start_samp, end_samp - WIN_SAMP + 1, HOP_SAMP):
            en = st + WIN_SAMP
            if en > y_len:
                continue
            if overlaps((st / SR, en / SR), exclusion_zones):
                continue

            seg = y[st:en]
            if len(seg) == WIN_SAMP:
                windows["Class_4"].append((file_id, st, seg))


# =========================================================
# REPORT COUNTS
# =========================================================
print("\nFinal window counts:")
for cls, items in windows.items():
    n_windows = len(items)
    n_files = len({x[0] for x in items})
    print(f"{cls}: {n_windows} windows | {n_files} unique file_ids")


# =========================================================
# SAVE WINDOWS
# =========================================================
OUT_BASE.mkdir(parents=True, exist_ok=True)
for folder in CLASS_FOLDERS.values():
    folder.mkdir(parents=True, exist_ok=True)

for cls, items in windows.items():
    out_dir = CLASS_FOLDERS[cls]
    for i, (file_id, start, audio) in enumerate(items):
        out_path = out_dir / f"{cls}_{int(file_id)}_{int(start)}_{i:04d}.wav"
        sf.write(out_path, audio, SR)

print(f"\nDone. Saved 0.2 s windows to: {OUT_BASE}")