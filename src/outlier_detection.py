import numpy as np
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt


def analyze_svm_outliers(svm_model, X, y, label_encoder, sigma_threshold=3.0, top_n=5):
    print("=" * 60)
    print("       ROBUST SVM OUTLIER DETECTION (Hybrid)       ")
    print("=" * 60)
    print(f"Detecting 3 types of points:")
    print("1. [Hard Sample]: High Slack, Normal Geometry (Valid)")
    print("2. [Label Suspect]: High Slack, Abnormal Geometry (Potential Label Error)")
    print("3. [Anomaly]: Low Slack, Abnormal Geometry (Corrupted Feature)\n")

    # 1. Pre-calculate Class Centroids & Std Devs (for Geometry)
    class_stats = {}
    global_hard_idx = []
    global_suspect_idx = []
    global_anomaly_idx = []
    for c in svm_model.classes_:
        X_c = X[y == c]
        centroid = np.mean(X_c, axis=0)
        # We use simple Euclidean distance to centroid
        dists = np.linalg.norm(X_c - centroid, axis=1)
        class_stats[c] = {
            "centroid": centroid,
            "mean_dist": np.mean(dists),
            "std_dist": np.std(dists),
        }

    # Iterate over classes (OvR)
    for c in svm_model.classes_:
        lbl_str = label_encoder.inverse_transform([int(c)])[0]
        print(f"\n--- Analyzing Class: {lbl_str} ---")

        clf = svm_model.models_[c]

        # --- A. Model-Based: Calculate Slack ---

        # Decision function: f(x)
        scores = clf.decision_function(X)

        # Slack: max(0, 1 - y*f(x))
        # We only care about positive slack (margin violations) for current class samples
        # Filter to only look at points belonging to this class (FN analysis)
        current_class_idx = np.where(y == c)[0]

        # Calculate slacks for these points
        margins = scores[current_class_idx]  # y is +1, so margin = score
        slacks = 1.0 - margins
        slacks[slacks < 0] = 0  # Ignore points safely beyond margin

        # --- B. Geometry-Based: Calculate Z-Score ---
        centroid = class_stats[c]["centroid"]
        mu_dist = class_stats[c]["mean_dist"]
        std_dist = class_stats[c]["std_dist"]

        dists = np.linalg.norm(X[current_class_idx] - centroid, axis=1)
        z_scores = (dists - mu_dist) / (std_dist + 1e-9)

        # --- C. Classification Logic ---
        # 1. Identify "High Slack" (
        #  slack > 1 (Misclassified)
        bad_slack_mask = slacks > 1.0

        # 2. Identify "Abnormal Geometry" (High Z-score)
        bad_geo_mask = z_scores > sigma_threshold

        # Combine
        local_suspects = np.where(bad_slack_mask & bad_geo_mask)[0]  # Label Noise
        local_hard = np.where(bad_slack_mask & ~bad_geo_mask)[0]  # Hard Samples
        local_anomaly = np.where(~bad_slack_mask & bad_geo_mask)[0]  # Feature Anomaly

        global_suspects = current_class_idx[local_suspects]
        global_hard = current_class_idx[local_hard]
        global_anomaly = current_class_idx[local_anomaly]

        # Extend Global Lists
        global_hard_idx.extend(global_hard)
        global_suspect_idx.extend(global_suspects)
        global_anomaly_idx.extend(global_anomaly)

        print(f"  COUNTS:")
        print(f"   - Zone A (Hard Samples):  {len(local_hard)}")
        print(f"   - Zone B (Suspects):      {len(local_suspects)}")
        print(f"   - Zone C (Anomalies):     {len(local_anomaly)}")

        def print_candidates(local_indices, title, limit=3):
            if len(local_indices) == 0:
                return
            # Sort by severity
            sort_metric = (
                z_scores[local_indices] if "Anomaly" in title else slacks[local_indices]
            )
            sorted_args = np.argsort(sort_metric)[::-1][:limit]

            print(f"  > Top {limit} {title}:")
            for i in sorted_args:
                li = local_indices[i]
                gi = current_class_idx[li]
                print(f"    idx={gi} | Slack={slacks[li]:.2f} | Z={z_scores[li]:.2f}")

        print_candidates(local_suspects, "Suspects (Review!)", limit=top_n)
        print_candidates(local_hard, "Hard Samples", limit=2)
        print_candidates(local_anomaly, "Anomalies", limit=2)

    # Convert to numpy arrays for plotting convenience
    global_hard_idx = np.array(global_hard_idx)
    global_suspect_idx = np.array(global_suspect_idx)
    global_anomaly_idx = np.array(global_anomaly_idx)

    # ==========================================================
    # VISUALIZATION
    # ==========================================================
    plt.figure(figsize=(12, 8))

    # 1. Plot Normal Data (Everything NOT in the special lists)
    all_outliers = (
        np.concatenate([global_hard_idx, global_suspect_idx, global_anomaly_idx])
        if (len(global_hard_idx) + len(global_suspect_idx) + len(global_anomaly_idx))
        > 0
        else np.array([])
    )

    normal_mask = np.ones(len(X), dtype=bool)
    if len(all_outliers) > 0:
        normal_mask[all_outliers.astype(int)] = False

    plt.scatter(
        X[normal_mask, 0],
        X[normal_mask, 1],
        c="lightgray",
        alpha=0.5,
        label="Normal Data",
        s=30,
    )

    # 2. Plot Zone A: Hard Samples (Blue Circles)
    if len(global_hard_idx) > 0:
        plt.scatter(
            X[global_hard_idx, 0],
            X[global_hard_idx, 1],
            c="blue",
            marker="o",
            s=50,
            label="Zone A: Hard Samples",
        )

    # 3. Plot Zone C: Anomalies (Green Triangles)
    if len(global_anomaly_idx) > 0:
        plt.scatter(
            X[global_anomaly_idx, 0],
            X[global_anomaly_idx, 1],
            c="green",
            marker="^",
            s=80,
            edgecolors="k",
            label="Zone C: Feature Anomalies",
        )

    # 4. Plot Zone B: Suspects (Red Crosses - The most important ones)
    if len(global_suspect_idx) > 0:
        plt.scatter(
            X[global_suspect_idx, 0],
            X[global_suspect_idx, 1],
            c="red",
            marker="X",
            s=100,
            linewidths=2,
            edgecolors="k",
            label="Zone B: Label Suspects",
        )

    plt.title("Robust SVM Outlier Detection (Slack vs Geometry)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return global_hard_idx, global_suspect_idx, global_anomaly_idx


def detect_cluster_outliers(X, kmeans_model, sigma_threshold=3.0):
    """
    Detects outliers using a statistical Z-score approach per cluster.
    Threshold = Mean_Distance + (sigma_threshold * Std_Dev_Distance)
    """
    print("\n" + "=" * 60)
    print(f"    CLUSTERING OUTLIER DETECTION (Z-Score > {sigma_threshold})")
    print("=" * 60)

    # 1. Get labels and centroids
    labels = kmeans_model.predict(X)
    centers = kmeans_model.cluster_centers_

    # 2. Calculate raw Euclidean distance of each point to its centroid
    distances = np.linalg.norm(X - centers[labels], axis=1)

    outlier_indices = []

    # 3. Iterate over each cluster
    unique_labels = np.unique(labels)

    for label in unique_labels:
        # Get points in this cluster
        cluster_indices = np.where(labels == label)[0]
        cluster_dists = distances[cluster_indices]

        mean_dist = np.mean(cluster_dists)
        std_dist = np.std(cluster_dists)

        # We flag points that are k standard deviations beyond the mean
        limit = mean_dist + (sigma_threshold * std_dist)

        local_outliers = np.where(cluster_dists > limit)[0]
        global_outliers = cluster_indices[local_outliers]
        outlier_indices.extend(global_outliers)

        # --- Print Top 3 for Inspection ---
        sorted_local_idx = np.argsort(cluster_dists)[::-1]
        top_3_idx = cluster_indices[sorted_local_idx[:3]]

        print(f"\n[Cluster {label}]")
        print(f"  Stats: Mean Dist={mean_dist:.2f}, Std={std_dist:.2f}")
        print(f"  Threshold (Mean + {sigma_threshold}*Std): {limit:.4f}")
        print(
            f"  Outliers Found: {len(global_outliers)} (out of {len(cluster_indices)} samples)"
        )
        print(f"  Top 3 Extreme Indices: {top_3_idx.tolist()}")

    outlier_indices = np.array(outlier_indices)

    # 4. Visualization
    plt.figure(figsize=(12, 8))

    # Plot Normal Data (Faint)
    normal_mask = np.ones(len(X), dtype=bool)
    if len(outlier_indices) > 0:
        normal_mask[outlier_indices] = False

    plt.scatter(
        X[normal_mask, 0],
        X[normal_mask, 1],
        c=labels[normal_mask],
        cmap="viridis",
        alpha=0.3,
        label="Normal Data",
    )

    # Plot Outliers (Red)
    if len(outlier_indices) > 0:
        plt.scatter(
            X[outlier_indices, 0],
            X[outlier_indices, 1],
            c="red",
            marker="x",
            s=100,
            linewidth=2,
            label=f"Outliers (> {sigma_threshold} std)",
        )

    plt.scatter(
        centers[:, 0],
        centers[:, 1],
        c="white",
        marker="o",
        edgecolor="black",
        s=200,
        linewidth=2,
        label="Centroids",
    )

    plt.title(
        f"Statistical Outlier Detection\n(Threshold = Mean + {sigma_threshold} * Std)"
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return outlier_indices
