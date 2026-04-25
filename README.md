# Multi-State PatchCore Audio

Anonymous code submission for industrial audio anomaly detection under real-world noise.

This repository contains the code used to evaluate a PatchCore-style anomaly detection pipeline for industrial machine acoustics and compare it against a supervised CNN14-based classification baseline.

The project focuses on short waveform windows extracted from annotated industrial audio recordings. Both pipelines operate on raw waveform windows. The CNN14/PANNs frontend computes log-Mel representations internally.

## Overview

The repository contains two main experimental pipelines:

1. **Supervised CNN baseline**

   A truncated CNN14 model is fine-tuned for 3-class supervised classification.

2. **Multi-State PatchCore-style retrieval pipeline**

   A pretrained CNN14 encoder is used only for feature extraction. No task-specific supervised training is performed for PatchCore. Class-specific memory banks are built for the three operating states, and test samples are classified by nearest-neighbor distance to the corresponding memory banks.

The three classes are:

| Class | Meaning |
|---|---|
| `Class_1` | Noisy background / operating condition |
| `Class_2` | Malfunction event |
| `Class_4` | Clean background / clean operating condition |

For the paper tables, both pipelines report:

| Metric | Description |
|---|---|
| Accuracy | Overall classification accuracy |
| Macro-F1 | Macro-averaged F1-score |
| FAR | False alarm rate: normal/background predicted as malfunction |
| ROC-AUC | Macro one-vs-rest ROC-AUC |
| PR-AUC | Macro one-vs-rest PR-AUC |

Both pipelines also report confusion matrices and classification reports.

---

## Repository Structure

```text
multi-state-patchcore-audio/
├── AudioDataset/
│   ├── Sample Audios/
│   └── Sample Annotations/
│
├── classification_pipeline/
│   ├── cnn14_truncated_finetune.py
│   ├── waveform_dataset.py
│   ├── train_cnn14_stage_unfreeze_weightedloss.py
│   └── evaluate_cnn14_stage_unfreeze.py
│
├── data_preprocessing/
│   ├── extract_0.2secWindows_3Classes.py
│   ├── extract_0.4secWindows_3Classes.py
│   ├── split_files_random.py
│   └── create_waveform_metadata_csv.py
│
├── external/
│   ├── __init__.py
│   ├── checkpoints/
│   │   └── class_labels_indices.csv
│   └── panns_inference/
│       ├── __init__.py
│       ├── config.py
│       ├── models.py
│       └── pytorch_utils.py
│
├── patchcore_pipeline/
│   ├── cnn14_truncated_feature_extractor.py
│   ├── extract_embeddings.py
│   ├── build_coresets.py
│   └── evaluate_3bank_knn.py
│
├── README.md
├── requirements.txt
└── .gitignore




## Setup and Usage

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The pretrained PANNs CNN14 checkpoint is not included in this repository. Place it manually under:

```text
external/checkpoints/Cnn14_mAP=0.431.pth
```

The repository includes a small anonymized sample dataset for code verification. The expected sample data structure is:

```text
AudioDataset/
├── Sample Audios/
│   ├── 121_S1.wav
│   ├── 350_S1.wav
│   └── ...
│
└── Sample Annotations/
    └── matching_df_one_event_per_row_3_subset.csv
```

The annotation CSV should contain at least `File_name`, `Class`, `Time_From_Reference`, and `Time_To_Reference`.

To create 0.2 s waveform windows, run:

```bash
python data_preprocessing/extract_0.2secWindows_3Classes.py \
  --wav_dir "AudioDataset/Sample Audios" \
  --csv_path "AudioDataset/Sample Annotations/matching_df_one_event_per_row_3_subset.csv" \
  --out_base outputs/sample_windows_0p2s_3class
```

To create 0.4 s waveform windows, run:

```bash
python data_preprocessing/extract_0.4secWindows_3Classes.py \
  --wav_dir "AudioDataset/Sample Audios" \
  --csv_path "AudioDataset/Sample Annotations/matching_df_one_event_per_row_3_subset.csv" \
  --out_base outputs/sample_windows_0p4s_3class
```

After extracting windows, create a file-level train/validation/test split and waveform metadata. For 0.2 s windows:

```bash
python data_preprocessing/split_files_random.py \
  --dataset_root outputs/sample_windows_0p2s_3class

python data_preprocessing/create_waveform_metadata_csv.py \
  --dataset_root outputs/sample_windows_0p2s_3class
```

For 0.4 s windows, use the same commands with `outputs/sample_windows_0p4s_3class`. The preprocessing creates `metadata_waveform.csv`, `train_metadata_waveform.csv`, `val_metadata_waveform.csv`, and `test_metadata_waveform.csv` under `File_split_CSVs/`. These waveform metadata files are used by both the classification and PatchCore pipelines.

## Supervised CNN Baseline

The supervised baseline uses a truncated CNN14 model:

```text
raw waveform
→ CNN14 spectrogram frontend
→ CNN14 log-Mel frontend
→ CNN14 conv_block1–3
→ global average pooling
→ embedding layer
→ classification head
```

To train the classifier on 0.2 s windows:

```bash
python classification_pipeline/train_cnn14_stage_unfreeze_weightedloss.py \
  --train_csv outputs/sample_windows_0p2s_3class/File_split_CSVs/train_metadata_waveform.csv \
  --val_csv outputs/sample_windows_0p2s_3class/File_split_CSVs/val_metadata_waveform.csv \
  --pann_checkpoint external/checkpoints/Cnn14_mAP=0.431.pth \
  --output_dir outputs/classification_0p2s
```

To evaluate the classifier on 0.2 s windows:

```bash
python classification_pipeline/evaluate_cnn14_stage_unfreeze.py \
  --test_csv outputs/sample_windows_0p2s_3class/File_split_CSVs/test_metadata_waveform.csv \
  --pann_checkpoint external/checkpoints/Cnn14_mAP=0.431.pth \
  --model_weights outputs/classification_0p2s/cnn14_stage_unfreeze_weighted_best.pth \
  --out_dir outputs/classification_0p2s/evaluation \
  --topk 5
```

For 0.4 s classification experiments, replace `0p2s` with `0p4s` and use `outputs/sample_windows_0p4s_3class` and `outputs/classification_0p4s`. The evaluation script reports both window-level and event/file-level results. Event-level aggregation uses Top-5 mean aggregation over the most confident windows per `file_id`.

## Multi-State PatchCore Pipeline

The PatchCore-style pipeline does not use supervised fine-tuning. It relies only on pretrained CNN14/PANNs weights:

```text
raw waveform
→ pretrained CNN14 frontend
→ pretrained conv_block1–3
→ global average pooling
→ 256-dimensional embedding
→ optional L2 normalization
→ class-specific memory banks
```

The pipeline builds three separate memory banks: `Class_1`, `Class_2`, and `Class_4`.

To extract PatchCore embeddings for 0.2 s windows:

```bash
python patchcore_pipeline/extract_embeddings.py \
  --dataset_root outputs/sample_windows_0p2s_3class \
  --pann_checkpoint external/checkpoints/Cnn14_mAP=0.431.pth \
  --output_dir outputs/patchcore_0p2s
```

Then build class-specific memory banks and k-center greedy coresets:

```bash
python patchcore_pipeline/build_coresets.py \
  --input_dir outputs/patchcore_0p2s \
  --k_coreset 1000
```

Then evaluate PatchCore:

```bash
python patchcore_pipeline/evaluate_3bank_knn.py \
  --input_dir outputs/patchcore_0p2s \
  --split test \
  --topk_neighbors 5 \
  --event_topk 5
```

For 0.4 s PatchCore experiments, replace `0p2s` with `0p4s` and use `outputs/sample_windows_0p4s_3class` and `outputs/patchcore_0p4s`. The PatchCore evaluation script reports both window-level and event/file-level metrics. At the event level, for each `file_id` and each class, the Top-5 smallest distances are averaged, and the predicted class is the class with the minimum aggregated distance.

## Event-Level Aggregation

Both pipelines report window-level and event/file-level results. For the supervised classifier, the event-level result is computed by selecting the Top-5 most confident windows per `file_id`, averaging their probability vectors, and predicting the class with the highest averaged probability. For PatchCore, the event-level result is computed by selecting the Top-5 smallest distances per class and `file_id`, averaging them, and predicting the class with the minimum aggregated distance. This provides a controlled comparison between window-level and event/file-level representations.

## Outputs and Notes

Generated outputs are written under `outputs/`, for example:

```text
outputs/
├── classification_0p2s/
│   └── evaluation/
│
└── patchcore_0p2s/
    ├── embeddings_full/
    ├── indices/
    ├── coresets/
    └── results/
```

Generated outputs, model weights, embeddings, and checkpoints are not tracked by Git. The full industrial dataset is not included in this repository. Only a small anonymized sample dataset is provided for code verification. The pretrained CNN14 checkpoint is also not included and must be placed manually at `external/checkpoints/Cnn14_mAP=0.431.pth`.

For anonymous review, author and institution information has been removed.
