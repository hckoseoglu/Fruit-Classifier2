import numpy as np
import matplotlib.pyplot as plt


class PCA_Scratch:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.eigenvalues_ = None

    def fit(self, X):
        # X should be (n_samples, n_features)
        n, d = X.shape

        # 1. Compute Mean and Center data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # 2. Compute Covariance Matrix
        # Note: We use (n-1) for sample covariance
        cov_matrix = (X_centered.T @ X_centered) / (n - 1)

        # 3. Eigen Decomposition
        # eigh returns eigenvalues in ascending order
        eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)

        # 4. Sort Descending (we want largest variance first)
        idx_sorted = np.argsort(eig_vals)[::-1]
        self.eigenvalues_ = eig_vals[idx_sorted]
        self.components_ = eig_vecs[:, idx_sorted].T  # Store vectors as rows

        # 5. Compute Explained Variance Ratio
        total_var = np.sum(self.eigenvalues_)
        self.explained_variance_ratio_ = self.eigenvalues_ / total_var

        return self

    def transform(self, X):
        X_centered = X - self.mean_
        # Project data: X_new = X_centered @ W^T
        # self.components_ is (n_components, n_features), so we transpose it
        k = self.n_components if self.n_components is not None else X.shape[1]
        return X_centered @ self.components_[:k].T

    def inverse_transform(self, X_transformed):
        k = X_transformed.shape[1]
        return (X_transformed @ self.components_[:k]) + self.mean_

def explore_intrinsic_dimensionality(X, variance_threshold=0.95, plot=True):
    """
    Applies PCA to explore intrinsic dimensionality via
    Explained Variance and Reconstruction Error.
    """
    print("=" * 60)
    print("       PCA INTRINSIC DIMENSIONALITY ANALYSIS       ")
    print("=" * 60)

    n_samples, n_features = X.shape

    # 1. Fit PCA on the full dimension
    pca = PCA_Scratch()
    pca.fit(X)

    eigenvalues = pca.eigenvalues_
    explained_var_ratio = pca.explained_variance_ratio_

    # 2. Cumulative Variance
    cumulative_var = np.cumsum(explained_var_ratio)

    # 3. Reconstruction Error (MSE)
    # The MSE when using k components is the sum of the eigenvalues of the
    # components we discard (from k+1 to d).
    # We calculate this efficiently by subtracting cumulative variance from total.
    total_variance = np.sum(eigenvalues)
    # retained_variance[k] = sum(lambda_1 ... lambda_k)
    retained_variance = np.cumsum(eigenvalues)
    # reconstruction_error[k] = Total - Retained
    reconstruction_error = total_variance - retained_variance
    # Handle the case of 0 components (error = total variance)
    reconstruction_error = np.insert(reconstruction_error, 0, total_variance)

    # --- Find "Elbow" and Thresholds ---

    # Dimensionality for 95% variance
    k_95 = np.argmax(cumulative_var >= variance_threshold) + 1

    # Intrinsic Dimensionality (Elbow method on Screen Plot)
    # A heuristic: looking for the point where the curve bends
    # We'll just report the top 5 components' variance

    print(f"Original Feature Dimension: {n_features}")
    print(f"Dimensions needed for {variance_threshold*100:.0f}% Variance: {k_95}")
    print(f"Reconstruction Error (MSE) at k={k_95}: {reconstruction_error[k_95]:.4f}")

    if plot:
      # --- Visualization ---
      fig, ax1 = plt.subplots(figsize=(10, 6))

      # Plot 1: Cumulative Explained Variance
      x_range = np.arange(1, n_features + 1)
      ax1.plot(
          x_range, cumulative_var, label="Cumulative Variance", color="blue", linewidth=2
      )
      ax1.axhline(
          y=variance_threshold,
          color="r",
          linestyle="--",
          alpha=0.7,
          label=f"{variance_threshold*100:.0f}% Threshold",
      )
      ax1.axvline(x=k_95, color="r", linestyle=":", alpha=0.7)

      ax1.set_xlabel("Number of Principal Components")
      ax1.set_ylabel("Cumulative Explained Variance Ratio", color="blue")
      ax1.tick_params(axis="y", labelcolor="blue")
      ax1.set_ylim(0, 1.05)
      ax1.grid(True, alpha=0.3)

      # Plot 2: Reconstruction Error (Twin Axis)
      ax2 = ax1.twinx()
      # We plot error starting from k=1
      ax2.plot(
          x_range,
          reconstruction_error[1:],
          label="Reconstruction Error (MSE)",
          color="orange",
          linewidth=2,
          linestyle="-.",
      )
      ax2.set_ylabel(
          "Reconstruction Error (Sum of Discarded Eigenvalues)", color="orange"
      )
      ax2.tick_params(axis="y", labelcolor="orange")

      # Title and Legend
      plt.title(
          f"Intrinsic Dimensionality: Variance vs Reconstruction Error\n(k={k_95} needed for {variance_threshold*100:.0f}%)"
      )

      # Combine legends
      lines1, labels1 = ax1.get_legend_handles_labels()
      lines2, labels2 = ax2.get_legend_handles_labels()
      ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

      plt.tight_layout()
      plt.show()

    return k_95, reconstruction_error
