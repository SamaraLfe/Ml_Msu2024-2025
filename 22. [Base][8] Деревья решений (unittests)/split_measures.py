import numpy as np


def evaluate_measures(sample):
    if not sample:
        return {'gini': 0.0, 'entropy': 0.0, 'error': 0.0}

    unique_classes, class_counts = np.unique(sample, return_counts=True)
    probabilities = class_counts / len(sample)
    gini = 1 - np.sum(np.square(probabilities))
    entropy = -np.sum(probabilities * np.log(probabilities))
    error = 1 - np.max(probabilities)
    measures = {'gini': gini, 'entropy': entropy, 'error': error}

    return measures