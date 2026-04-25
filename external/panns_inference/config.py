import csv
from pathlib import Path

sample_rate = 32000


# config.py is inside external/panns_inference/.
# Therefore, parents[1] points to external/.
EXTERNAL_DIR = Path(__file__).resolve().parents[1]
labels_csv_path = EXTERNAL_DIR / "checkpoints" / "class_labels_indices.csv"

if not labels_csv_path.exists():
    raise FileNotFoundError(
        f"Missing class label file: {labels_csv_path}\n"
        "Please place class_labels_indices.csv under external/checkpoints/."
    )

labels = []
ids = []

with open(labels_csv_path, "r", newline="", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter=",")
    lines = list(reader)

for i1 in range(1, len(lines)):
    label_id = lines[i1][1]
    label = lines[i1][2]
    ids.append(label_id)
    labels.append(label)

classes_num = len(labels)

lb_to_ix = {label: i for i, label in enumerate(labels)}
ix_to_lb = {i: label for i, label in enumerate(labels)}

id_to_ix = {label_id: i for i, label_id in enumerate(ids)}
ix_to_id = {i: label_id for i, label_id in enumerate(ids)}