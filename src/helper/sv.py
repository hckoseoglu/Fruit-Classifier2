import numpy as np

# ------------------------------------------------------------
# 1. Extract margin support vectors per class
# ------------------------------------------------------------
def get_margin_sv_sets(svm, X_train_s, eps=1e-6):
    """
    Returns:
      dict[class_id] -> array of margin support vectors (n_sv, d)
    """
    sv_sets = {}
    for c in svm.classes_:
        clf = svm.models_[c]
        a = clf.alpha_
        mask = (a > eps) & (a < clf.C - eps)   # margin SVs
        sv_sets[int(c)] = X_train_s[mask]
    return sv_sets


# ------------------------------------------------------------
# 2. Directional kNN distance between two SV sets
# ------------------------------------------------------------
def knn_set_distance(A, B, k=5):
    """
    Directional distance A -> B using average kNN Euclidean distance.
    """
    if A.size == 0 or B.size == 0:
        return np.nan

    k_eff = min(k, B.shape[0])

    # squared Euclidean distances
    A2 = np.sum(A * A, axis=1, keepdims=True)      # (na, 1)
    B2 = np.sum(B * B, axis=1, keepdims=True).T    # (1, nb)
    D2 = A2 + B2 - 2.0 * (A @ B.T)
    D2 = np.maximum(D2, 0.0)

    # k nearest neighbors for each row
    idx = np.argpartition(D2, kth=k_eff - 1, axis=1)[:, :k_eff]
    knn_d = np.sqrt(np.take_along_axis(D2, idx, axis=1))

    return float(np.mean(knn_d))


# ------------------------------------------------------------
# 3. SV kNN distance matrix (THIS is what you asked about)
# ------------------------------------------------------------
def sv_knn_distance_matrix(
    svm,
    X_train_s,
    k=5,
    eps=1e-6,
    use_margin_only=True,
    symmetrize=True
):
    """
    Returns:
      classes : array of class ids
      D       : (C x C) SV distance matrix
    """
    classes = np.array(svm.classes_, dtype=int)
    sv_sets = get_margin_sv_sets(svm, X_train_s, eps=eps)

    C = len(classes)
    D = np.zeros((C, C), dtype=float)

    for i, ci in enumerate(classes):
        for j, cj in enumerate(classes):
            if i == j:
                D[i, j] = 0.0
            else:
                D[i, j] = knn_set_distance(
                    sv_sets[int(ci)],
                    sv_sets[int(cj)],
                    k=k
                )

    if symmetrize:
        for i in range(C):
            for j in range(i + 1, C):
                D[i, j] = D[j, i] = 0.5 * (D[i, j] + D[j, i])

    return classes, D


# ------------------------------------------------------------
# 4. Distance → similarity conversion
# ------------------------------------------------------------
def distance_to_similarity(D, mode="inverse"):
    """
    Convert distance matrix to similarity matrix.
    """
    if mode == "inverse":
        return 1.0 / (1.0 + D)
    elif mode == "exp":
        return np.exp(-D)
    else:
        raise ValueError("mode must be 'inverse' or 'exp'")
