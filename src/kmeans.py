from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linear_sum_assignment
import numpy as np

def analyze_clustering(X, y_true, k=5, seed=42):
    """
    Applies K-Means and evaluates using Inertia, Silhouette, and Purity.
    """
    print("\n" + "="*60)
    print("       CLUSTERING ANALYSIS (Scikit-Learn)       ")
    print("="*60)

    
    print(f"Running K-Means with k={k}...")

    # 2. Fit K-Means
    # n_init='auto' usually runs 10 times to find best initialization
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init='auto')
    y_pred = kmeans.fit_predict(X)

    # -------------------------------------------------------
    # INTERNAL METRICS (No Ground Truth needed)
    # -------------------------------------------------------
    
    # A. Inertia (Sum of squared distances to closest centroid)
    # Lower is better (but naturally decreases as k increases)
    inertia = kmeans.inertia_
    
    # B. Silhouette Score
    # Range: -1 (wrong cluster) to +1 (perfectly dense & separated)
    # 0 implies overlapping clusters.
    sil_score = silhouette_score(X, y_pred)

    print(f"\n[Internal Metrics]")
    print(f"Inertia (SSE):     {inertia:.4f}")
    print(f"Silhouette Score:  {sil_score:.4f}")

    # -------------------------------------------------------
    # EXTERNAL METRICS (Comparing to Ground Truth)
    # -------------------------------------------------------
    
    # C. Purity
    # Logic: For each cluster, find the most frequent true class. 
    # Sum these counts and divide by total samples.
    
    # Compute contingency matrix (confusion matrix)
    cm = confusion_matrix(y_true, y_pred)
    # For each column (cluster), take the max value (dominant class count)
    majority_sum = np.sum(np.amax(cm, axis=0))
    purity = majority_sum / len(y_true)

    print(f"\n[External Metrics]")
    print(f"Purity Score:      {purity:.4f}")

    # -------------------------------------------------------
    # VISUALIZATION (Scatter Plot on first 2 dimensions)
    # -------------------------------------------------------
    # We plot the first 2 dimensions. If X is the PCA-reduced data, 
    # this will effectively be PC1 vs PC2.
    
    plt.figure(figsize=(10, 6))
    
    # Scatter plot colored by CLUSTER ID
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', alpha=0.6, edgecolor='k', s=50)
    
    # Plot Centroids (White X)
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='white', marker='X', s=200, edgecolor='black', linewidth=2, label='Centroids')
    
    plt.title(f'K-Means Clustering (k={k})\nPurity: {purity:.2f} | Silhouette: {sil_score:.2f}')
    plt.xlabel('Feature 1 (or PC1)')
    plt.ylabel('Feature 2 (or PC2)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return kmeans, y_pred


def plot_elbow_curve(X, max_k=10, seed=42):
    print("\n" + "="*60)
    print("       SSE ELBOW ANALYSIS       ")
    print("="*60)
    
    sse_values = []
    k_range = range(1, max_k + 1)
    
    for k in k_range:
        # n_init='auto' ensures we pick the best of several runs to minimize SSE
        kmeans = KMeans(n_clusters=k, random_state=seed, n_init='auto')
        kmeans.fit(X)
        sse_values.append(kmeans.inertia_) # Inertia IS SSE
        
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, sse_values, marker='o', linestyle='-', color='b')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Sum of Squared Error (Inertia)')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True, alpha=0.3)
    plt.xticks(k_range)
    plt.show()
    
    return sse_values

