import numpy as np

def get_part_of_array(X: np.ndarray) -> np.ndarray:
    return X[::4, 120:500:5]

def sum_non_neg_diag(X: np.ndarray) -> int:
    diag = np.diag(X)
    non_neg_diag = diag[diag >= 0]
    return non_neg_diag.sum() if non_neg_diag.size > 0 else -1

def replace_values(X: np.ndarray) -> np.ndarray:
    X_copy = X.copy()
    M = np.mean(X, axis=0)
    mask = (X < 0.25 * M) | (X > 1.5 * M)
    X_copy[mask] = -1
    return X_copy