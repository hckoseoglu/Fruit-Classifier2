import time
import json
import numpy as np
import pickle
from cvxopt import matrix, solvers
import pprint
from interpret_svm import (
    visualize_class_ovr_with_farthest_all_classes,
    get_sv_and_extremes_for_class,
    summarize_point,
    summarize_sv,
)
from visualization.histogram_svm import plot_margin_histogram_ovr
from outlier_detection import analyze_svm_outliers, detect_cluster_outliers
from visualization.matrix_heatmap import plot_matrix_heatmap
from helper.confusion_matrix import compute_confusion_matrix, normalize_confusion_matrix
from helper.sv import sv_knn_distance_matrix, distance_to_similarity
from pca import explore_intrinsic_dimensionality, PCA_Scratch
from kmeans import analyze_clustering, plot_elbow_curve
from helper.split import stratified_train_test_split, get_original_data_points
from helper.metrics import accuracy, weighted_f1
from helper.standardizer import Standardizer

EXTRACTED_DATASET_FILE = "./dataset_with_outliers_reduced.json"
PROCESSORS_FILE = "./feature_processors.pkl"
USE_PCA = True


# ----------------------------
# Binary Linear SVM via QP
# ----------------------------


class BinaryLinearSVM_QP:
    def __init__(self, C=1.0, eps=1e-6):
        self.C = float(C)
        self.eps = float(eps)

    def fit(self, X, y):
        """
        X: (n, d) float
        y: (n,) in {-1, +1}
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, d = X.shape

        # Gram matrix for linear kernel
        K = X @ X.T  # (n, n)

        # QP: minimize (1/2) a^T P a + q^T a
        # where P = (y y^T) .* K, q = -1
        P = (y[:, None] * y[None, :]) * K
        q = -np.ones(n)

        # Constraints: 0 <= a_i <= C
        G = np.vstack([-np.eye(n), np.eye(n)])
        h = np.hstack([np.zeros(n), np.ones(n) * self.C])

        # Equality: y^T a = 0
        A = y.reshape(1, -1)
        b = np.array([0.0])

        # CVXOPT expects 'matrix' doubles
        P_cvx = matrix(P)
        q_cvx = matrix(q)
        G_cvx = matrix(G)
        h_cvx = matrix(h)
        A_cvx = matrix(A)
        b_cvx = matrix(b)

        # make solver quiet
        solvers.options["show_progress"] = False

        sol = solvers.qp(P_cvx, q_cvx, G_cvx, h_cvx, A_cvx, b_cvx)
        a = np.array(sol["x"]).reshape(-1)

        self.alpha_ = a

        # Support vectors
        sv_mask = a > self.eps
        self.sv_idx_ = np.where(sv_mask)[0]

        # Compute w
        self.w_ = (a * y) @ X  # (d,)

        # Compute b using margin SVs (0 < alpha < C)
        margin_mask = (a > self.eps) & (a < self.C - self.eps)
        margin_idx = np.where(margin_mask)[0]

        if len(margin_idx) > 0:
            b_vals = y[margin_idx] - (X[margin_idx] @ self.w_)
            self.b_ = float(np.mean(b_vals))
        else:
            # fallback: use any support vector
            if len(self.sv_idx_) == 0:
                # degenerate case (shouldn't happen with proper data/C)
                self.b_ = 0.0
            else:
                i = self.sv_idx_[0]
                self.b_ = float(y[i] - X[i] @ self.w_)

        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float64)
        return X @ self.w_ + self.b_

    def predict(self, X):
        scores = self.decision_function(X)
        return np.where(scores >= 0, 1, -1)

    def support_vectors_indices(self):
        return self.sv_idx_

    def hyperplane_norm(self):
        return float(np.linalg.norm(self.w_) + 1e-12)


# ----------------------------
# Multi-class OvR using the binary SVM above
# ----------------------------


class LinearSVM_OvR_QP:
    def __init__(self, C=1.0, eps=1e-6):
        self.C = C
        self.eps = eps

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.models_ = {}
        for c in self.classes_:
            y_bin = np.where(y == c, 1.0, -1.0)
            clf = BinaryLinearSVM_QP(C=self.C, eps=self.eps)
            clf.fit(X, y_bin)
            self.models_[c] = clf
        return self

    def decision_function(self, X):
        # scores: (n_samples, n_classes)
        scores = np.column_stack(
            [self.models_[c].decision_function(X) for c in self.classes_]
        )
        return scores

    def predict(self, X):
        scores = self.decision_function(X)
        # choose class with largest score
        return self.classes_[np.argmax(scores, axis=1)]

    def support_vectors_per_class(self):
        # return original indices *within the training matrix* for each OvR classifier
        return {c: self.models_[c].support_vectors_indices() for c in self.classes_}

    def farthest_points_per_class(self, X, y):
        """
        For each class c (its OvR hyperplane):
          - farthest positive: max signed distance among y==c
          - farthest negative: min signed distance among y!=c
        Distance = (w^T x + b) / ||w||
        Returns dict c -> {"pos": idx, "neg": idx}  (indices w.r.t X)
        """
        out = {}
        for c in self.classes_:
            clf = self.models_[c]
            normw = clf.hyperplane_norm()
            dist = clf.decision_function(X) / normw

            pos_idx = np.where(y == c)[0]
            neg_idx = np.where(y != c)[0]

            pos_far = pos_idx[np.argmax(dist[pos_idx])] if len(pos_idx) else None
            neg_far = neg_idx[np.argmin(dist[neg_idx])] if len(neg_idx) else None

            out[c] = {"pos": pos_far, "neg": neg_far}
        return out


# ----------------------------
# dataset loading
# ----------------------------


with open(EXTRACTED_DATASET_FILE, "r") as f:
    data = json.load(f)

text_feats = np.array([entry["text_features"] for entry in data], dtype=np.float64)
numeric_feats = np.array(
    [entry["numerical_features"] for entry in data], dtype=np.float64
)
cat_feats = np.array(
    [entry["categorical_features"] for entry in data], dtype=np.float64
)
image_feats = np.array([entry["image_features"] for entry in data], dtype=np.float64)
y_encoded = np.array([entry["label_encoded"] for entry in data])

with open(PROCESSORS_FILE, "rb") as f:
    processors = pickle.load(f)
label_encoder = processors["label_encoder"]

X = np.concatenate([text_feats, numeric_feats, cat_feats, image_feats], axis=1)

# Split first
X_train, X_test, y_train, y_test, train_idx, test_idx = stratified_train_test_split(
    X, y_encoded, test_size=0.2, seed=42
)

# Standardize before PCA
scaler = Standardizer()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

if USE_PCA:
    # PCA to reduce dimensionality before SVM (on standardized data)
    k_optimal, errors = explore_intrinsic_dimensionality(
        X_train_s, variance_threshold=0.95, plot=False
    )
    print(f"Using PCA with k={k_optimal} components to retain 95% variance.")
    pca = PCA_Scratch(n_components=k_optimal)
    pca.fit(X_train_s)
    X_train_s = pca.transform(X_train_s)
    X_test_s = pca.transform(X_test_s)

# Train OvR SVM-QP
svm = LinearSVM_OvR_QP(C=1.0, eps=1e-6)

t0 = time.perf_counter()
svm.fit(X_train_s, y_train)
train_time = time.perf_counter() - t0
print(f"SVM(QP) training time: {train_time:.3f} s")

# Evaluate
y_pred = svm.predict(X_test_s)
print("Test accuracy:", accuracy(y_test, y_pred))
print("Test weighted F1:", weighted_f1(y_test, y_pred))

# Support vectors (per OvR classifier; indices are within X_train_s)
sv_per_class = svm.support_vectors_per_class()
for c in svm.classes_:
    print(f"class {c}: #support vectors = {len(sv_per_class[c])}")

# Farthest points from each OvR hyperplane (indices are within X_train_s)
farthest = svm.farthest_points_per_class(X_train_s, y_train)
for c in svm.classes_:
    fp = farthest[c]
    print(f"class {c}: farthest pos idx={fp['pos']}  farthest neg idx={fp['neg']}")


#### This part is for comparing support vectors and farthest points with other models ####


# block sizes from your original arrays (only valid if not using PCA)
if not USE_PCA:
    d_text = text_feats.shape[1]
    d_num = numeric_feats.shape[1]
    d_cat = cat_feats.shape[1]
    d_img = image_feats.shape[1]

    class_id = int(np.unique(y_train)[0])  # pick one class to inspect first

    info = get_sv_and_extremes_for_class(svm, X_train_s, y_train, class_id, top_k=5)

    print("Class:", class_id)
    print("#Support vectors:", len(info["sv_idx"]))
    print("Mean |dist| of SVs:", float(np.mean(np.abs(info["dist"][info["sv_idx"]]))))

    print("\nFarthest positive point:")
    pprint.pprint(
        summarize_point(
            info["far_pos"],
            X_train_s,
            y_train,
            d_text,
            d_num,
            d_cat,
            d_img,
            info["dist"],
        )
    )

    print("\nFarthest negative point:")
    pprint.pprint(
        summarize_point(
            info["far_neg"],
            X_train_s,
            y_train,
            d_text,
            d_num,
            d_cat,
            d_img,
            info["dist"],
        )
    )

    print("\nClosest-to-hyperplane points (ambiguous):")
    for idx in info["closest"]:
        pprint.pprint(
            summarize_point(
                idx, X_train_s, y_train, d_text, d_num, d_cat, d_img, info["dist"]
            )
        )
else:
    print("\nSkipping block norm analysis (PCA transformed features)")

# Pick the OvR classifier for class 0
c = 0
clf = svm.models_[c]  # BinaryLinearSVM_QP for class c vs rest

# Build binary labels for this OvR problem: +1 for class c, -1 otherwise
y_bin = np.where(y_train == c, 1.0, -1.0)

# Decision values f(x) = w^T x + b
f = clf.decision_function(X_train_s)

# Functional margins m = y * f
m = y_bin * f

# Support vectors via alpha threshold
eps = 1e-6
a = clf.alpha_
sv_mask = a > eps

# Margin SVs vs slack SVs
margin_sv_mask = (a > eps) & (a < clf.C - eps)
slack_sv_mask = a >= (clf.C - eps)

print("Class", c)
print("#SV:", np.sum(sv_mask))
print("#margin SV (0<alpha<C):", np.sum(margin_sv_mask))
print("#slack SV (alpha≈C):", np.sum(slack_sv_mask))


# Summaries
summarize_sv("m on SV", m[sv_mask])
summarize_sv("m on margin SV", m[margin_sv_mask])
summarize_sv("m on slack SV", m[slack_sv_mask])

tol = 1e-4
print("#SV inside margin (m<1):", np.sum(m[sv_mask] < 1.0 - tol))
print("#SV misclassified (m<0):", np.sum(m[sv_mask] < 0.0 + tol))

# Visualize support vectors and furthest points
# Create histogram for functional margion of these points
visualize_class_ovr_with_farthest_all_classes(
    svm, X_train_s, y_train, class_id=4, label_encoder=label_encoder
)

plot_margin_histogram_ovr(
    svm=svm,
    X_train_s=X_train_s,
    y_train=y_train,
    class_id=4,  # orange
    label_encoder=label_encoder,
)

## This part is for comparing SV similarities and confusion matrix ##

# SV kNN distance matrix
classes, D = sv_knn_distance_matrix(
    svm=svm, X_train_s=X_train_s, k=5, eps=1e-6, use_margin_only=True, symmetrize=True
)

# Convert to similarity (optional but useful)
S = distance_to_similarity(D, mode="inverse")


# Raw confusion matrix
C_raw = compute_confusion_matrix(y_true=y_test, y_pred=y_pred, classes=classes)

# Row-normalized confusion matrix
C_row = normalize_confusion_matrix(C_raw, mode="row")

plot_matrix_heatmap(
    M=D,
    classes=classes,
    label_encoder=label_encoder,
    title="Support Vector kNN Distance Matrix",
    cmap="magma",
    fmt="{:.2f}",
)

plot_matrix_heatmap(
    M=S,
    classes=classes,
    label_encoder=label_encoder,
    title="Support Vector kNN Similarity Matrix",
    cmap="viridis",
    fmt="{:.3f}",
)


plot_matrix_heatmap(
    M=C_row,
    classes=classes,
    label_encoder=label_encoder,
    title="Confusion Matrix",
    cmap="Blues",
    fmt="{:.2f}",
)


#### This part is for outlier detection analysis ####

(hard, suspect, anomaly) = analyze_svm_outliers(
    svm_model=svm,
    X=X_train_s,
    y=y_train,
    label_encoder=label_encoder,
    sigma_threshold=1.5,
    top_n=3,
)

suspect_points, suspect_orig_indices = get_original_data_points(suspect, train_idx)
anomaly_points, anomaly_orig_indices = get_original_data_points(anomaly, train_idx)
# print number of hard samples, suspect points, anomaly points
print(f"Number of hard samples: {len(hard)}")
print(f"Number of suspect points: {len(suspect)}")
print(f"Number of anomaly points: {len(anomaly)}")

number_of_anomalies_ai = 0
with open("outliers_detected.txt", "w") as f:
    f.write("SUSPECT POINTS FROM SVM-BASED OUTLIER DETECTION\n")
    f.write("=" * 60 + "\n\n")
    for point in suspect_points:
        f.write(pprint.pformat(point) + "\n\n")

    f.write("\n" + "=" * 60 + "\n")
    f.write("ANOMALY POINTS FROM SVM-BASED OUTLIER DETECTION (with 'ai' in img)\n")
    f.write("=" * 60 + "\n\n")
    for point in anomaly_points:
        if "ai" in point["img"]:
            number_of_anomalies_ai += 1
            f.write(pprint.pformat(point) + "\n\n")

print("\nOutliers saved to outliers_detected.txt")
print(
    f"Number of anomaly points with 'ai': {number_of_anomalies_ai} out of {len(anomaly)}"
)

# Pca analysis
X_train_pca = X_train_s

if USE_PCA:
    # Already applied PCA at the beginning, use transformed data for clustering
    print("\nUsing PCA-transformed features for clustering analysis")
else:
    # Apply PCA for clustering analysis only
    k_optimal, errors = explore_intrinsic_dimensionality(
        X_train_s, variance_threshold=0.95
    )

    pca_opt = PCA_Scratch(n_components=k_optimal)
    pca_opt.fit(X_train_s)
    X_train_pca = pca_opt.transform(X_train_s)


# Clustering analysis

kmeans_model, clusters = analyze_clustering(
    X=X_train_pca,
    y_true=y_train,  # For calculating Purity
    k=len(np.unique(y_train)),
    seed=42,
)

sse_scores = plot_elbow_curve(X_train_pca, max_k=10)

# Outlier detection based on clustering
outliers_z = detect_cluster_outliers(X_train_pca, kmeans_model, sigma_threshold=3.0)

# Compare with SVM-based outliers
# anomalies plus suspect points from SVM analysis
svm_outlier_set = set(anomaly).union(set(suspect))
cluster_outlier_set = set(outliers_z)
# Print overlaps size
overlap = svm_outlier_set.intersection(cluster_outlier_set)
print("\nOutlier Detection Overlap:")
print(f"SVM-based outliers (anomaly+suspect): {len(svm_outlier_set)}")
print(f"Clustering-based outliers: {len(cluster_outlier_set)}")
print(f"Overlap count: {len(overlap)}")
