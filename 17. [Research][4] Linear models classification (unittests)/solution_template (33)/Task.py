import numpy as np


class Preprocessor:

    def __init__(self):
        pass

    def fit(self, X, Y=None):
        pass

    def transform(self, X):
        pass

    def fit_transform(self, X, Y=None):
        pass


class MyOneHotEncoder(Preprocessor):

    def __init__(self, dtype=np.float64):
        super().__init__()
        self.encoding_table, self.offset_table, self.new_features_total_size = None, None, 0
        self.dtype = dtype

    def fit(self, X, Y=None):
        self.encoding_table = {}
        self.offset_table = {}
        self.new_features_total_size = 0
        for feature in X:
            unique_vals = sorted(X[feature].unique())
            self.encoding_table[feature] = {}
            for idx, value in enumerate(unique_vals):
                self.encoding_table[feature][value] = idx
            self.offset_table[feature] = self.new_features_total_size
            self.new_features_total_size += len(unique_vals)

    def transform(self, X):
        total = np.zeros((X.shape[0], self.new_features_total_size), dtype=int)
        for feature in X:
            if feature in self.offset_table and feature in self.encoding_table:
                for i, value in enumerate(X[feature]):
                    if value in self.encoding_table[feature]:
                        j = self.offset_table[feature] + self.encoding_table[feature][value]
                        total[i][j] = 1
        return total

    def fit_transform(self, X, Y=None):
        self.fit(X)
        return self.transform(X)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


class SimpleCounterEncoder:

    def __init__(self, dtype=np.float64):
        self.dtype = dtype

    def fit(self, X, Y):
        self.successes = {}
        self.counters = {}
        for feature in X:
            self.successes[feature] = {}
            self.counters[feature] = {}
            unique_vals = X[feature].unique()
            for val in unique_vals:
                self.successes[feature][val] = np.mean(Y.loc[X[feature] == val])
                self.counters[feature][val] = len(Y.loc[X[feature] == val]) / X.shape[0]

    def transform(self, X, a=1e-5, b=1e-5):
        total = np.zeros((X.shape[0], X.shape[1] * 3), dtype=self.dtype)
        for j, feature in enumerate(X):
            for i, value in enumerate(X[feature]):
                success_val = self.successes[feature].get(value, 0)
                counter_val = self.counters[feature].get(value, 0)
                total[i][3*j] = success_val
                total[i][3*j + 1] = counter_val
                total[i][3*j + 2] = (success_val + a) / (counter_val + b) if counter_val != 0 else 0
        return total

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


def group_k_fold(size, n_splits=3, seed=1):
    idx = np.arange(size)
    np.random.seed(seed)
    idx = np.random.permutation(idx)
    n_ = size // n_splits
    for i in range(n_splits - 1):
        yield idx[i * n_: (i + 1) * n_], np.hstack((idx[:i * n_], idx[(i + 1) * n_:]))
    yield idx[(n_splits - 1) * n_:], idx[:(n_splits - 1) * n_]


class FoldCounters:

    def __init__(self, n_folds=3, dtype=np.float64):
        self.dtype = dtype
        self.n_folds = n_folds

    def fit(self, X, Y, seed=1):
        self.folds = list(group_k_fold(X.shape[0], self.n_folds, seed))
        self.successes, self.counters = [{} for _ in self.folds], [{} for _ in self.folds]

        for i, fold in enumerate(self.folds):
            X_, Y_ = X.iloc[fold[1]], Y.iloc[fold[1]]
            for feature in X_:
                unique_vals = X_[feature].unique()
                self.successes[i][feature] = {val: np.mean(Y_[X_[feature] == val]) for val in unique_vals}
                self.counters[i][feature] = {val: len(Y_[X_[feature] == val]) / X_.shape[0] for val in unique_vals}

    def transform(self, X, a=1e-5, b=1e-5):
        tot = X.shape[0], X.shape[1] * 3
        total = np.zeros(tot, dtype=np.float64)
        for c, fold in enumerate(self.folds):
            for i in fold[0]:
                for j, feature in enumerate(X.columns):
                    val = X.iloc[i, j]
                    success = self.successes[c][feature][val]
                    counter = self.counters[c][feature][val]
                    total[i, 3*j:3*j+3] = [success, counter, (success + a) / (counter + b)]
        return total

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)


def weights(x, y):
    encoding_table = {val: idx for idx, val in enumerate(np.unique(x))}
    c = (x.shape[0], len(encoding_table))
    masse = np.zeros(c, dtype=int)

    for i, val in np.ndenumerate(x):
        masse[i][encoding_table[val]] = 1

    sum0 = np.sum(masse * (y[:, None] == 0), axis=0)
    sum1 = np.sum(masse * (y[:, None] == 1), axis=0)
    total = sum1 / (sum1 + sum0)

    return total