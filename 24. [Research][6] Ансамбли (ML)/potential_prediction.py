import os
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline
import numpy as np


class PotentialTransformer:
    """
    A potential transformer.

    This class is used to convert the potential's 2d matrix to 1d vector of features.
    """

    def fit(self, x, y):
        """
        Build the transformer on the training set.
        :param x: list of potential's 2d matrices
        :param y: target values (can be ignored)
        :return: trained transformer
        """
        return self

    def fit_transform(self, x, y):
        """
        Build the transformer on the training set and return the transformed dataset (1d vectors).
        :param x: list of potential's 2d matrices
        :param y: target values (can be ignored)
        :return: transformed potentials (list of 1d vectors)
        """
        return self.transform(x)

    def transform(self, x):
        """
        Transform the list of potential's 2d matrices with the trained transformer.
        :param x: list of potential's 2d matrices
        :return: transformed potentials (list of 1d vectors)
        """
        transformed_data = []
        for matrix in x:
            matrix -= 20
            non_zero_coords = np.argwhere(matrix != 0)
            if non_zero_coords.size == 0:
                transformed_data.append(matrix.flatten())
                continue

            min_coords = non_zero_coords.min(axis=0)
            max_coords = non_zero_coords.max(axis=0)
            center_non_zero = (min_coords + max_coords) // 2
            center_arr = np.array(matrix.shape) // 2
            shift = center_arr - center_non_zero

            result = np.zeros_like(matrix)
            new_coords = non_zero_coords + shift

            for coord, new_coord in zip(non_zero_coords, new_coords):
                if all(0 <= n < dim for n, dim in zip(new_coord, matrix.shape)):
                    result[tuple(new_coord)] = matrix[tuple(coord)]

            transformed_data.append(result.flatten())
        return np.array(transformed_data)


def load_dataset(data_dir):
    """
    Read potential dataset.

    This function reads dataset stored in the folder and returns three lists
    :param data_dir: the path to the potential dataset
    :return:
    files -- the list of file names
    np.array(X) -- the list of potential matrices (in the same order as in files)
    np.array(Y) -- the list of target value (in the same order as in files)
    """
    files, X, Y = [], [], []
    for file in sorted(os.listdir(data_dir)):
        potential = np.load(os.path.join(data_dir, file))
        files.append(file)
        X.append(potential["data"])
        Y.append(potential["target"])
    return files, np.array(X), np.array(Y)


def train_model_and_predict(train_dir, test_dir):
    _, X_train, Y_train = load_dataset(train_dir)
    test_files, X_test, _ = load_dataset(test_dir)
    potential_transformer = PotentialTransformer()
    X_train = potential_transformer.fit_transform(X_train, Y_train)
    X_test = potential_transformer.fit_transform(X_test, Y_train)
    regressor = Pipeline([('vectorizer', potential_transformer),
                          ('decision_tree',
                           ExtraTreesRegressor(n_estimators=3000, criterion="friedman_mse", max_depth=10,
                                               max_features="sqrt", n_jobs=-1))])
    regressor.fit(X_train, Y_train)
    predictions = regressor.predict(X_test)
    return {file: value for file, value in zip(test_files, predictions)}