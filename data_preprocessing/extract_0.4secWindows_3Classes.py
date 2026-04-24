"""
extract_0.4secWindows_3Classes.py
---------------------------------

This script generates fixed-length waveform windows of 0.4 s from long audio
recordings using event-level annotations provided in a CSV file.

We construct three classes:

- Class_1: noisy background / environmental operating condition
- Class_2: malfunction events, typically short and transient
- Class_4: clean background / clean machine operation

For Class_1 and Class_4, we extract sliding windows only within the annotated
time intervals. To avoid contamination from malfunction events, we exclude
windows that overlap with expanded Class_2 exclusion zones.

For Class_2, we generate centered windows around the annotated malfunction
intervals. Because malfunction events are short and temporally localized, we
also include small deterministic temporal offsets (jitter) around the event
center to improve robustness to minor alignment differences.

All valid windows are retained. No class balancing is applied at this stage.

Input WAV directory:
    AudioDataset/Sample Audios/

Input annotation CSV:
    AudioDataset/Sample Annotations/matching_df_one_event_per_row_3_subset.csv

Output directory:
    outputs/sample_windows_0p4s_3class/
        Class_1/
        Class_2/
        Class_4/
"""

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf


# ============================================================
# PATH CONFIGURATION
# ============================================================

parser = argparse.ArgumentParser(
    description="Generate fixed-length 0.4 s waveform windows from annotated audio."
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
    default=Path("outputs/sample_windows_0p4s_3class"),
    help="Output directory for extracted 0.4 s windows.",
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


# ============================================================
# AUDIO AND WINDOW CONFIGURATION
# ============================================================

# We resample all recordings to a fixed sampling rate to ensure
# consistent temporal resolution across files.
SR = 25600

# Window configuration: 0.4 s = 10240 samples at 25.6 kHz.
WIN_SEC = 0.4
WIN_SAMP = int(WIN_SEC * SR)

# Sliding-window hop size for Class_1 and Class_4.
# A hop of 0.2 s corresponds to 50% overlap.
HOP_SEC = 0.2
HOP_SAMP = int(HOP_SEC * SR)

# We enlarge malfunction intervals by a small temporal margin so that
# background windows do not accidentally include malfunction content.
EXCLUDE_MARGIN = 0.10  # seconds

# For Class_2, we generate multiple windows around the event center
# using small deterministic temporal offsets.
K_JITTER = 3


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_wav_file(file_number: int) -> Path | None:
    """
    Locate the waveform file associated with a given file identifier.

    Files are expected to follow the naming pattern:
        <number>_S1.wav
        <number>_S2.wav

    We prefer the S1 version when available. If neither S1 nor S2 exists,
    we search for any matching file with the same prefix.
    """
    f1 = WAV_DIR / f"{int(file_number)}_S1.wav"
    if f1.exists():
        return f1

    f2 = WAV_DIR / f"{int(file_number)}_S2.wav"
    if f2.exists():
        return f2

    matches = sorted(WAV_DIR.glob(f"{int(file_number)}_S*.wav"))
    return matches[0] if matches else None


def clamp_centered_window(center_sec: float, signal_len: int) -> tuple[int, int]:
    """
    Construct a centered 0.4 s window around a given time point.

    The window is clamped to valid signal boundaries so that the extracted
    segment always remains within the waveform.

    Args:
        center_sec: Center time in seconds.
        signal_len: Length of the waveform in samples.

    Returns:
        (start_sample, end_sample)
    """
    center_sample = int(center_sec * SR)
    start = center_sample - WIN_SAMP // 2
    start = max(0, min(start, signal_len - WIN_SAMP))
    end = start + WIN_SAMP
    return start, end


def overlaps(
    interval: tuple[float, float],
    forbidden_intervals: list[tuple[float, float]],
) -> bool:
    """
    Check whether a candidate interval overlaps with any forbidden interval.

    Args:
        interval: Tuple (start_sec, end_sec)
        forbidden_intervals: List of tuples (start_sec, end_sec)

    Returns:
        True if overlap exists, otherwise False.
    """
    a, b = interval
    for s, e in forbidden_intervals:
        if not (b <= s or a >= e):
            return True
    return False


# ============================================================
# LOAD ANNOTATION CSV
# ============================================================

df = pd.read_csv(CSV_PATH)

required_cols = {"File_name", "Class", "Time_From_Reference", "Time_To_Reference"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"CSV missing required columns: {missing}")

# We store extracted windows separately for each class.
windows = {
    "Class_1": [],
    "Class_2": [],
    "Class_4": [],
}


# ============================================================
# WINDOW EXTRACTION PER FILE
# ============================================================

for file_id, group in df.groupby("File_name"):

    wav_path = find_wav_file(file_id)
    if wav_path is None:
        print(f"[WARN] Missing wav file for File_name={file_id}")
        continue

    # Load waveform in mono at the target sampling rate.
    y, _ = librosa.load(wav_path, sr=SR, mono=True)
    y_len = len(y)

    # --------------------------------------------------------
    # Build exclusion zones from Class_2 annotations
    # --------------------------------------------------------
    # We expand each malfunction interval slightly to avoid using
    # nearby samples as clean background windows.
    class2_rows = group[group["Class"] == "Class_2"]
    exclusion_zones = []

    for _, row in class2_rows.iterrows():
        s = float(row["Time_From_Reference"]) - EXCLUDE_MARGIN
        e = float(row["Time_To_Reference"]) + EXCLUDE_MARGIN
        exclusion_zones.append((s, e))

    # --------------------------------------------------------
    # Extract Class_2 windows: centered + temporal jitter
    # --------------------------------------------------------
    # Malfunction events are short and localized. We therefore center
    # windows on the event midpoint and add small deterministic offsets.
    for _, row in class2_rows.iterrows():
        center = (
            float(row["Time_From_Reference"]) +
            float(row["Time_To_Reference"])
        ) / 2.0

        max_jitter = min(0.10, WIN_SEC / 4)
        offsets = np.linspace(-max_jitter, max_jitter, K_JITTER)

        for off in offsets:
            st, en = clamp_centered_window(center + float(off), y_len)
            seg = y[st:en]

            if len(seg) == WIN_SAMP:
                windows["Class_2"].append((file_id, st, seg))

    # --------------------------------------------------------
    # Extract Class_1 windows: sliding windows within annotation
    # --------------------------------------------------------
    for _, row in group[group["Class"] == "Class_1"].iterrows():
        s = float(row["Time_From_Reference"])
        e = float(row["Time_To_Reference"])

        start_samp = int(s * SR)
        end_samp = int(e * SR)

        for st in range(start_samp, end_samp - WIN_SAMP + 1, HOP_SAMP):
            en = st + WIN_SAMP

            if en > y_len:
                continue

            # Reject windows that overlap with malfunction exclusion zones.
            if overlaps((st / SR, en / SR), exclusion_zones):
                continue

            seg = y[st:en]
            if len(seg) == WIN_SAMP:
                windows["Class_1"].append((file_id, st, seg))

    # --------------------------------------------------------
    # Extract Class_4 windows: sliding windows within annotation
    # --------------------------------------------------------
    for _, row in group[group["Class"] == "Class_4"].iterrows():
        s = float(row["Time_From_Reference"])
        e = float(row["Time_To_Reference"])

        start_samp = int(s * SR)
        end_samp = int(e * SR)

        for st in range(start_samp, end_samp - WIN_SAMP + 1, HOP_SAMP):
            en = st + WIN_SAMP

            if en > y_len:
                continue

            # Reject windows that overlap with malfunction exclusion zones.
            if overlaps((st / SR, en / SR), exclusion_zones):
                continue

            seg = y[st:en]
            if len(seg) == WIN_SAMP:
                windows["Class_4"].append((file_id, st, seg))


# ============================================================
# KEEP ALL VALID WINDOWS
# ============================================================
# No balancing is applied here. The goal of this extraction step is
# to preserve all valid windows for downstream splitting and modeling.

print("\nFinal window counts to save (all valid windows retained):")
for cls, items in windows.items():
    n_windows = len(items)
    n_files = len(set(x[0] for x in items))
    print(f"{cls}: {n_windows} windows | from {n_files} unique file_ids")


# ============================================================
# SAVE EXTRACTED WINDOWS
# ============================================================

OUT_BASE.mkdir(parents=True, exist_ok=True)
for folder in CLASS_FOLDERS.values():
    folder.mkdir(parents=True, exist_ok=True)

for cls, items in windows.items():
    out_dir = CLASS_FOLDERS[cls]
    for i, (file_id, start, audio) in enumerate(items):
        out_path = out_dir / f"{cls}_{int(file_id)}_{int(start)}_{i:04d}.wav"
        sf.write(out_path, audio, SR)

print("\nDone. Saved 0.4 s waveform windows to:", OUT_BASE)