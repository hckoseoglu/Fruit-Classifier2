import numpy as np
import json


def get_original_data_points(
    indices, train_idx, dataset_file="./dataset_with_outliers_reduced.json"
):
   
    with open(dataset_file, "r") as f:
        data = json.load(f)

    original_indices = train_idx[indices]
    original_entries = [data[int(i)] for i in original_indices]

    print(f"\nOriginal dataset indices: {original_indices.tolist()}")

    return original_entries, original_indices


def stratified_train_test_split(X, y, test_size=0.2, seed=42):
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    train_idx = []
    test_idx = []

    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_test = int(np.ceil(len(idx) * test_size))
        test_idx.append(idx[:n_test])
        train_idx.append(idx[n_test:])

    train_idx = np.concatenate(train_idx)
    test_idx = np.concatenate(test_idx)

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx], train_idx, test_idx
