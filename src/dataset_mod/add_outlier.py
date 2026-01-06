import pickle
import json
import numpy as np

PROCESSORS = "./feature_processors.pkl"
DATASET = "./dataset_ai_extracted_reduced.json"
NEW_DATASET = "./dataset_with_outliers_reduced.json"

labels = ["banana", "lemon", "mandarin", "orange", "apple"]
with open(PROCESSORS, "rb") as f:
    processors = pickle.load(f)
label_encoder = processors["label_encoder"]

labels_encoded = label_encoder.transform(labels)


# Scan dataset.json and for each class parse first N samples,
# copy them, flip their label to a different class, and save them
# as outliers.json
def add_outliers_to_dataset(
    dataset_json_path,
    outliers_json_path,
    n_per_class=5,
):

    with open(dataset_json_path, "r") as f:
        data = json.load(f)

    outliers = []
    class_to_indices = {cls: [] for cls in labels_encoded}

    # Collect indices per class
    for idx, entry in enumerate(data):
        lbl = entry["label_encoded"]
        if lbl in class_to_indices and len(class_to_indices[lbl]) < n_per_class:
            class_to_indices[lbl].append(idx)

    # Copy original data
    outliers.extend(data)

    # Create outliers by flipping labels
    for cls, indices in class_to_indices.items():
        for idx in indices:
            entry = data[idx].copy()
            label = entry["category"]
            new_label = labels[(labels.index(label) + 1) % len(labels)]
            entry["category"] = new_label
            new_cls = label_encoder.transform([new_label])[0]
            entry["label_encoded"] = int(new_cls)
            entry["source"] = "add_outlier.py"
            outliers.append(entry)
            index_of_entry = len(outliers) - 1
            print(
                f"Added outlier: Original Index {idx}, New Index {index_of_entry}, Original Class '{label}' -> New Class '{new_label}'"
            )

    with open(outliers_json_path, "w") as f:
        json.dump(outliers, f, indent=2)


add_outliers_to_dataset(
    dataset_json_path=DATASET,
    outliers_json_path=NEW_DATASET,
    n_per_class=5,
)
