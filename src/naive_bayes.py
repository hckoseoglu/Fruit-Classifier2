import time
import json
import numpy as np
import pickle

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from pca import PCA_Scratch, explore_intrinsic_dimensionality

## GLOBALS ##
EXTRACTED_DATASET_FILE = "./dataset_with_outliers_reduced.json"
PROCESSORS_FILE = "./feature_processors.pkl"
USE_PCA = True
###########################

# Load dataset
with open(EXTRACTED_DATASET_FILE, "r") as f:
    data = json.load(f)

text_feats = np.array([entry["text_features"] for entry in data])
numeric_feats = np.array([entry["numerical_features"] for entry in data])
cat_feats = np.array([entry["categorical_features"] for entry in data])
image_feats = np.array([entry["image_features"] for entry in data])
y_encoded = np.array([entry["label_encoded"] for entry in data])

with open(PROCESSORS_FILE, "rb") as f:
    processors = pickle.load(f)
label_encoder = processors["label_encoder"]

X = np.concatenate([text_feats, numeric_feats, cat_feats, image_feats], axis=1)

# Split first
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    shuffle=True,
    stratify=y_encoded,
    random_state=40,
)

# Standardize features (always do this before PCA)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Apply PCA if enabled
if USE_PCA:
    print("Applying PCA to reduce dimensionality...")
    k_optimal, errors = explore_intrinsic_dimensionality(
        X_train_s, variance_threshold=0.95, plot=False
    )
    print(f"Optimal PCA components to retain 95% variance: {k_optimal}")

    pca = PCA_Scratch(n_components=k_optimal)
    pca.fit(X_train_s)
    X_train_s = pca.transform(X_train_s)
    X_test_s = pca.transform(X_test_s)
    print(f"Data shape after PCA: {X_train_s.shape}")

model = Pipeline([("clf", GaussianNB(var_smoothing=1e-9))])

t0 = time.perf_counter()
model.fit(X_train_s, y_train)
train_time = time.perf_counter() - t0
print(f"GaussianNB training time: {train_time:.6f} s")

test_pred = model.predict(X_test_s)

train_acc = model.score(X_train_s, y_train)
test_acc = model.score(X_test_s, y_test)
test_f1 = f1_score(y_test, test_pred, average="weighted")

print(f"Training Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")
