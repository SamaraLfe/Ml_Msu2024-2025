from typing import List
from copy import deepcopy

def get_part_of_array(X: List[List[float]]) -> List[List[float]]:
    return [X[i][120:500:5] for i in range(0, len(X), 4)]

def sum_non_neg_diag(X: List[List[int]]) -> int:
    n = min(len(X), len(X[0]))
    if any(X[i][i] >= 0 for i in range(n)):
        return sum(X[i][i] for i in range(n))
    else:
        return -1

def replace_values(X: List[List[float]]) -> List[List[float]]:
    X_copy = deepcopy(X)
    n = len(X)
    for j in range(len(X[0])):
        column_sum = sum(X[i][j] for i in range(n))
        M = column_sum / n
        for i in range(n):
            if X[i][j] < 0.25 * M or X[i][j] > 1.5 * M:
                X_copy[i][j] = -1
    return X_copy