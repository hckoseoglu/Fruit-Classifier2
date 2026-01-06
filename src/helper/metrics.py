import numpy as np

def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def weighted_f1(y_true, y_pred):
    classes = np.unique(y_true)
    f1_sum = 0.0
    n = len(y_true)

    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        support = np.sum(y_true == c)
        f1_sum += (support / n) * f1

    return float(f1_sum)