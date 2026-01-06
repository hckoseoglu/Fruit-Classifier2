import numpy as np
import matplotlib.pyplot as plt

def plot_margin_histogram_ovr(
    svm,
    X_train_s,
    y_train,
    class_id,
    label_encoder,
    eps=1e-6,
    tol=1e-4,
    bins=60
):
    """
    Plot functional margin histogram for OvR classifier of class_id.
    Shows:
      - all points
      - support vectors
      - (optional) slack SVs
      - margin threshold m=1
      - decision boundary m=0
    """

    clf = svm.models_[class_id]
    class_name = label_encoder.inverse_transform([class_id])[0]

    # OvR binary labels
    y_bin = np.where(y_train == class_id, 1.0, -1.0)

    # Functional margin
    f = clf.decision_function(X_train_s)
    m = y_bin * f

    # Alphas
    a = clf.alpha_

    sv_mask = a > eps
    margin_sv_mask = (a > eps) & (a < clf.C - eps)
    slack_sv_mask = a >= (clf.C - eps)

    # ---- Plot ----
    plt.figure(figsize=(9, 5))

    plt.hist(
        m,
        bins=bins,
        alpha=0.45,
        label="All training points"
    )

    plt.hist(
        m[sv_mask],
        bins=bins,
        alpha=0.75,
        label="Support vectors"
    )

    if np.any(slack_sv_mask):
        plt.hist(
            m[slack_sv_mask],
            bins=bins,
            alpha=0.85,
            label="Slack support vectors"
        )

    # Reference lines
    plt.axvline(1.0, linestyle=":", linewidth=2, label="Margin (m = 1)")
    plt.axvline(0.0, linestyle="--", linewidth=2, label="Decision boundary (m = 0)")
    plt.axvline(1.0 - tol, linestyle=":", linewidth=1, color="gray",
                label=f"Numerical tol (1 - {tol})")

    plt.title(f"Functional margin distribution — OvR({class_name})")
    plt.xlabel(r"$m = y(w^T x + b)$")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---- Print summary ----
    print(f"=== Margin summary for OvR({class_name}) ===")
    print(f"C = {clf.C}")
    print(f"#SV total: {np.sum(sv_mask)}")
    print(f"#Margin SVs: {np.sum(margin_sv_mask)}")
    print(f"#Slack SVs: {np.sum(slack_sv_mask)}")
    print(f"#Misclassified (m < 0): {np.sum(m < 0)}")
    print(f"#Inside margin (m < 1 - tol): {np.sum((a > eps) & (m < 1 - tol))}")
    print(f"m on SVs: min={m[sv_mask].min():.6f}, "
          f"mean={m[sv_mask].mean():.6f}, "
          f"max={m[sv_mask].max():.6f}, "
          f"std={m[sv_mask].std():.6f}")
