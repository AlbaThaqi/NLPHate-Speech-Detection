"""Training utilities (example stub)

This is a minimal example using scikit-learn to train a logistic regression.
"""
from sklearn.linear_model import LogisticRegression
from joblib import dump


def train_logistic(X, y, model_path: str = "models/logistic.joblib"):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    dump(clf, model_path)
    return clf
