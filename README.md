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
