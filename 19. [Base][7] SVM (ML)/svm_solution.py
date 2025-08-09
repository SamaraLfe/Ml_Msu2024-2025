import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


def train_svm_and_predict(train_features, train_target, test_features):

    model_svm = SVC(kernel="rbf", C=2.5, gamma="auto", class_weight="balanced")
    scaler = StandardScaler()
    scaler.fit(train_features)
    model_svm.fit(scaler.transform(train_features), train_target)
    predictions = model_svm.predict(scaler.transform(test_features))
    return predictions