from collections import Counter
from typing import List

def are_multisets_equal(x: List[int], y: List[int]) -> bool:
    return Counter(x) == Counter(y)

def max_prod_mod_3(x: List[int]) -> int:
    return max((x[i] * x[i + 1] for i in range(len(x) - 1) if x[i] % 3 == 0 or x[i + 1] % 3 == 0), default=-1)

def convert_image(image: List[List[List[float]]], weights: List[float]) -> List[List[float]]:
    return [[sum(p * w for p, w in zip(pixel, weights)) for pixel in row] for row in image]

def rle_scalar(x: List[List[int]], y:  List[List[int]]) -> int:
    x_new = [value for value, count in x for _ in range(count)]
    y_new = [value for value, count in y for _ in range(count)]
    return sum(a * b for a, b in zip(x_new, y_new)) if len(x_new) == len(y_new) else -1

def cosine_distance(X: List[List[float]], Y: List[List[float]]) -> List[List[float]]:
    x1 = [sum(x * x for x in vec) ** 0.5 for vec in X]
    y1 = [sum(y * y for y in vec) ** 0.5 for vec in Y]
    x2 = {i for i, val in enumerate(x1) if val == 0}
    y2 = {i for i, val in enumerate(y1) if val == 0}

    res = [[1.0 if i in x2 else (0 if j in y2 else sum(x * y for x, y in zip(X[i], Y[j])) / (x1[i] * y1[j]))
          for j in range(len(Y))]
          for i in range(len(X))]
    return res