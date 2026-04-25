"""
waveform_dataset.py
-------------------

PyTorch dataset for loading raw waveform windows from metadata CSV files.

Each sample returns:
- waveform: torch.FloatTensor of shape (samples,)
- label: torch.LongTensor scalar

The metadata input can be either:
- a path to a metadata CSV file
- a pandas DataFrame

The metadata is expected to contain at least:
- path
- label

The "path" column may point either to:
1) a real .wav file
2) a log-mel .npy file

If the path points to a log-mel .npy file, the corresponding waveform path is
reconstructed by replacing the parent folder suffix:

Example:
    Class_4_log-mels/Class_4_62_49449_1491.npy
    ->
    Class_4/Class_4_62_49449_1491.wav
"""

from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset


class WaveformDataset(Dataset):
    """
    Dataset that loads raw waveform windows from metadata.

    Args:
        metadata:
            Path to a metadata CSV file or a pandas DataFrame.
        sample_rate:
            Expected waveform sampling rate.
    """

    def __init__(self, metadata, sample_rate=25600):
        if isinstance(metadata, (str, Path)):
            self.df = pd.read_csv(metadata)
        elif isinstance(metadata, pd.DataFrame):
            self.df = metadata.reset_index(drop=True)
        else:
            raise TypeError("metadata must be a CSV path or a pandas DataFrame")

        self.sample_rate = sample_rate
        self.paths = self.df["path"].values
        self.labels = self.df["label"].values

    def __len__(self):
        return len(self.paths)

    def _resolve_wav_path(self, path_value) -> Path:
        """
        Resolve the waveform path from a metadata path.

        If the metadata path already points to a .wav file, it is returned.
        If it points to a .npy log-mel file, the path is mapped back to the
        corresponding .wav file.
        """
        p = Path(path_value)

        # Case 1: metadata already points to a waveform file
        if p.suffix.lower() == ".wav":
            return p

        # Case 2: metadata points to a log-mel .npy file
        if p.suffix.lower() == ".npy":
            parent_name = p.parent.name

            # Example:
            # Class_4_log-mels -> Class_4
            if parent_name.endswith("_log-mels"):
                wav_parent = p.parent.parent / parent_name.replace("_log-mels", "")
            else:
                wav_parent = p.parent

            return wav_parent / f"{p.stem}.wav"

        # Fallback: keep path as-is
        return p

    def __getitem__(self, idx):
        raw_path = self.paths[idx]
        wav_path = self._resolve_wav_path(raw_path)

        try:
            wave, sr = sf.read(str(wav_path))
        except Exception as e:
            file_id = self.df.iloc[idx]["file_id"] if "file_id" in self.df.columns else "N/A"
            raise RuntimeError(
                f"Failed to read waveform at idx={idx}\n"
                f"metadata_path={raw_path}\n"
                f"resolved_wav_path={wav_path}\n"
                f"label={self.labels[idx]}\n"
                f"file_id={file_id}\n"
                f"original_error={repr(e)}"
            ) from e

        if sr != self.sample_rate:
            raise ValueError(
                f"Expected {self.sample_rate}Hz, got {sr} for file {wav_path}"
            )

        wave = torch.tensor(wave, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return wave, label