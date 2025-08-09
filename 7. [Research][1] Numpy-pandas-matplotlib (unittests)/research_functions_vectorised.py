import numpy as np

def are_multisets_equal(x: np.ndarray, y: np.ndarray) -> bool:
    return np.array_equal(np.sort(x), np.sort(y))

def max_prod_mod_3(x: np.ndarray) -> int:
    prod = np.prod([x[:-1], x[1:]], axis=0)
    mod_3 = prod[np.mod(prod, 3) == 0]
    return np.max(mod_3) if mod_3.size else -1

def convert_image(image: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.tensordot(image, weights, axes=([2], [0]))

def rle_scalar(x: np.ndarray, y: np.ndarray) -> int:
    x_new = np.repeat(x[:, 0], x[:, 1])
    y_new = np.repeat(y[:, 0], y[:, 1])
    return np.dot(x_new, y_new) if len(x_new) == len(y_new) else -1

def cosine_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    x1 = np.linalg.norm(X, axis=1)
    y1 = np.linalg.norm(Y, axis=1)
    mul = np.outer(x1, y1).astype(np.float64)
    m = np.dot(X, Y.T).astype(np.float64)
    res = np.divide(m, mul, out=np.ones_like(m, dtype=np.float64), where=mul != 0)
    return res