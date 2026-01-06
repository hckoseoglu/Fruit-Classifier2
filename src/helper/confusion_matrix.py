import numpy as np

def compute_confusion_matrix(y_true, y_pred, classes):
    """
    Compute confusion matrix.

    Args:
      y_true   : array of true labels (encoded)
      y_pred   : array of predicted labels (encoded)
      classes  : array of class ids (e.g. svm.classes_)

    Returns:
      C : (C x C) confusion matrix where
          C[i, j] = # true class i predicted as j
    """
    
    # check if y_true and y_pred have all elements same
    # this way we check if the model has any error
    if not np.array_equal(y_true, y_pred):
        print("Warning: y_true and y_pred are not identical. There are misclassifications.")
    else:
        print("y_true and y_pred are identical. No misclassifications detected.")
    
    classes = np.array(classes, dtype=int)
    C = len(classes)
    idx = {c: i for i, c in enumerate(classes)}

    M = np.zeros((C, C), dtype=int)

    for yt, yp in zip(y_true, y_pred):
        i = idx[int(yt)]
        j = idx[int(yp)]
        M[i, j] += 1

    return M


def normalize_confusion_matrix(M, mode="row"):
    """
    Normalize confusion matrix.

    mode="row": rows sum to 1 (P(pred=j | true=i))
    mode="all": divide by total samples
    """
    M = M.astype(float)

    if mode == "row":
        row_sums = M.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return M / row_sums
    elif mode == "all":
        return M / np.sum(M)
    else:
        raise ValueError("mode must be 'row' or 'all'")