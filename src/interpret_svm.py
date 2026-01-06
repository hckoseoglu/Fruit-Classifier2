import numpy as np
import pickle
import matplotlib.pyplot as plt

PROCESSORS_FILE = "./feature_processors.pkl"

# ---------- Load label encoder ----------
with open(PROCESSORS_FILE, "rb") as f:
    processors = pickle.load(f)
label_encoder = processors["label_encoder"]


# ---------- Helpers ----------
def pca_2d(X):
    """2D PCA projection via SVD (no sklearn)."""
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ Vt[:2].T
    return Z


def ovr_scores(svm, X):
    """Scores shape (n_samples, n_classes) in svm.classes_ order."""
    return np.column_stack([svm.models_[c].decision_function(X) for c in svm.classes_])


def ovr_predict(svm, X):
    scores = ovr_scores(svm, X)
    return svm.classes_[np.argmax(scores, axis=1)]


def get_sv_and_extremes_for_class(
    ovr_model, X_train_s, y_train, class_id, eps=1e-6, top_k=10
):
    clf = ovr_model.models_[class_id]
    d = signed_distance(clf, X_train_s)

    # support vectors (alpha > eps)
    sv_idx = np.where(clf.alpha_ > eps)[0]

    # farthest positive among true class samples
    pos_idx = np.where(y_train == class_id)[0]
    far_pos = pos_idx[np.argmax(d[pos_idx])] if len(pos_idx) else None

    # farthest negative among non-class samples
    neg_idx = np.where(y_train != class_id)[0]
    far_neg = neg_idx[np.argmin(d[neg_idx])] if len(neg_idx) else None

    # closest points to hyperplane (most ambiguous)
    closest = np.argsort(np.abs(d))[:top_k]

    return {
        "sv_idx": sv_idx,
        "far_pos": far_pos,
        "far_neg": far_neg,
        "closest": closest,
        "dist": d,
    }


def signed_distance(clf, X):
    w = clf.w_
    b = clf.b_
    normw = np.linalg.norm(w) + 1e-12
    return (X @ w + b) / normw


def block_norms(x, d_text, d_num, d_cat, d_img):
    i0 = 0
    # normalize with number of features in each block
    t = np.linalg.norm(x[i0 : i0 + d_text]) / np.sqrt(d_text)
    i0 += d_text
    n = np.linalg.norm(x[i0 : i0 + d_num]) / np.sqrt(d_num)
    i0 += d_num
    c = np.linalg.norm(x[i0 : i0 + d_cat]) / np.sqrt(d_cat)
    i0 += d_cat
    im = np.linalg.norm(x[i0 : i0 + d_img]) / np.sqrt(d_img)
    total_norm = np.linalg.norm(x) / np.sqrt(d_text + d_num + d_cat + d_img)
    return t, n, c, im, total_norm


def summarize_point(idx, X_train_s, y_train, d_text, d_num, d_cat, d_img, dist=None):
    x = X_train_s[idx]
    t, n, c, im, total_norm = block_norms(x, d_text, d_num, d_cat, d_img)
    out = {
        "idx": int(idx),
        "true_class": label_encoder.inverse_transform([int(y_train[idx])])[0],
        "dist_to_hyperplane": float(dist[idx]) if dist is not None else None,
        "norm_text": float(t),
        "norm_num": float(n),
        "norm_cat": float(c),
        "norm_img": float(im),
        "total_norm": float(total_norm),
    }
    return out


def summarize_sv(name, arr):
    if arr.size == 0:
        print(name, "EMPTY")
        return
    print(
        f"{name}: min={arr.min():.6f}, mean={arr.mean():.6f}, max={arr.max():.6f}, std={arr.std():.6f}"
    )


def visualize_class_ovr_with_farthest_all_classes(
    svm,
    X_train_s,
    y_train,
    class_id,
    label_encoder,
    eps=1e-6,
    max_points=4000,
    seed=42,
    annotate_farthest=True,
):
    """
    PCA 2D scatter of training data.
    Focus: ONE OvR classifier for `class_id`
      - margin SVs: black circles (hollow)
      - slack SVs (alpha≈C): orange circles (hollow)
      - misclassified SVs (m<0): red circles (hollow)
    Also: farthest positive correctly classified point for EACH class (stars).
    Prints counts of SV types at the end.
    """
    rng = np.random.default_rng(seed)
    n = X_train_s.shape[0]

    # Subsample for background points if large
    if n > max_points:
        idx_plot = rng.choice(n, size=max_points, replace=False)
    else:
        idx_plot = np.arange(n)

    Xp = X_train_s[idx_plot]
    yp = y_train[idx_plot]

    # PCA projection
    Z = pca_2d(Xp)

    # Label names (decoded)
    class_ids_all = np.unique(y_train)
    class_names_all = label_encoder.inverse_transform(class_ids_all)

    # For multiclass correctness of farthest points
    y_pred_train = ovr_predict(svm, X_train_s)

    # ---- Focus classifier (class_id vs rest) ----
    clf = svm.models_[class_id]
    a = clf.alpha_

    # Support vectors
    sv_mask = a > eps
    margin_sv_mask = (a > eps) & (a < clf.C - eps)
    slack_sv_mask = a >= (clf.C - eps)

    # Misclassified SVs for this OvR task: m = y_bin * f < 0
    y_bin = np.where(y_train == class_id, 1.0, -1.0)
    f = clf.decision_function(X_train_s)
    m = y_bin * f

    mis_sv_mask = sv_mask & (m < 0)

    sv_idx = np.where(sv_mask)[0]
    margin_sv_idx = np.where(margin_sv_mask)[0]
    slack_sv_idx = np.where(slack_sv_mask)[0]
    mis_sv_idx = np.where(mis_sv_mask)[0]

    # ---- Farthest positive (correctly classified) for EACH class ----
    farthest_pos_idx = {}  # class_id -> idx in full training set
    
    for c in svm.classes_:
        clf_c = svm.models_[c]
        d_c = signed_distance(clf_c, X_train_s)
        # candidate = true class c AND predicted c (correct overall)
        cand = np.where((y_train == c) & (y_pred_train == c))[0]
        if len(cand) == 0:
            farthest_pos_idx[c] = None
        else:
            idx_max = int(cand[np.argmax(d_c[cand])]) 
            print("Farthest positive for class", c, "is index", idx_max)
            farthest_pos_idx[c] = idx_max
        

    # ---- Map full indices to plotted indices ----
    idx_plot_map = {int(i): j for j, i in enumerate(idx_plot)}

    def to_plot_indices(idxs_full):
        return np.array(
            [idx_plot_map[i] for i in idxs_full if int(i) in idx_plot_map], dtype=int
        )

    margin_sv_plot = to_plot_indices(margin_sv_idx)
    slack_sv_plot = to_plot_indices(slack_sv_idx)
    mis_sv_plot = to_plot_indices(mis_sv_idx)

    farthest_plot = []
    farthest_class = []
    for c, idx_full in farthest_pos_idx.items():
        if idx_full is None:
            continue
        if idx_full in idx_plot_map:
            farthest_plot.append(idx_plot_map[idx_full])
            farthest_class.append(c)
    


    # ---- Plot ----
    plt.figure(figsize=(10, 7))

    # Background: true class colors
    for cid, cname in zip(class_ids_all, class_names_all):
        mask = yp == cid
        plt.scatter(Z[mask, 0], Z[mask, 1], s=12, alpha=0.45, label=str(cname))

    focus_name = label_encoder.inverse_transform([class_id])[0]

    # Margin SVs (black hollow)
    if margin_sv_plot.size > 0:
        plt.scatter(
            Z[margin_sv_plot, 0],
            Z[margin_sv_plot, 1],
            s=80,
            facecolors="none",
            edgecolors="black",
            linewidths=1.3,
            label=f"Margin SVs (OvR: {focus_name})",
        )

    # Slack SVs (orange hollow)
    if slack_sv_plot.size > 0:
        plt.scatter(
            Z[slack_sv_plot, 0],
            Z[slack_sv_plot, 1],
            s=90,
            facecolors="none",
            edgecolors="orange",
            linewidths=1.6,
            label=f"Slack SVs (OvR: {focus_name})",
        )

    # Misclassified SVs (red hollow) — draw last so they’re visible
    if mis_sv_plot.size > 0:
        plt.scatter(
            Z[mis_sv_plot, 0],
            Z[mis_sv_plot, 1],
            s=100,
            facecolors="none",
            edgecolors="red",
            linewidths=2.0,
            label=f"Misclassified SVs (OvR: {focus_name})",
        )

    # Farthest positive correctly classified for each class (stars)
    for p_idx, c in zip(farthest_plot, farthest_class):
        cname = label_encoder.inverse_transform([c])[0]
        plt.scatter(
            Z[p_idx, 0],
            Z[p_idx, 1],
            s=230,
            marker="*",
            edgecolors="black",
            linewidths=1.1,
            label=f"Farthest + (correct): {cname}",
        )
        if annotate_farthest:
            plt.annotate(
                cname,
                (Z[p_idx, 0], Z[p_idx, 1]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=9,
            )
    
    plt.title(
        f"PCA-2D: SV types for OvR({focus_name}) + farthest correct points (all classes)"
    )   
    cname = label_encoder.inverse_transform([c])[0]
    
    if annotate_farthest:
        plt.annotate(
            cname,
            (Z[p_idx, 0], Z[p_idx, 1]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
        )

    plt.title(
        f"PCA-2D: SV types for OvR({focus_name}) + farthest correctly-classified positives (all classes)"
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    # De-duplicate legend labels
    handles, labels = plt.gca().get_legend_handles_labels()
    seen = set()
    new_h, new_l = [], []
    for h, lab in zip(handles, labels):
        if lab not in seen:
            seen.add(lab)
            new_h.append(h)
            new_l.append(lab)
    plt.legend(new_h, new_l, loc="best", fontsize=9)

    plt.tight_layout()
    plt.show()

    print(f"=== OvR sanity for class {focus_name} (id={class_id}) ===")
    print(f"C = {clf.C}")
    print(f"#SV total (alpha>eps): {len(sv_idx)}")
    print(f"#Margin SVs (0<alpha<C): {len(margin_sv_idx)}")
    print(f"#Slack SVs (alpha≈C): {len(slack_sv_idx)}")
    print(f"#Misclassified SVs (m<0): {len(mis_sv_idx)}")
