"""Prediction helper (stub)"""
from joblib import load


def predict_texts(model_path: str, X):
    model = load(model_path)
    return model.predict(X)
