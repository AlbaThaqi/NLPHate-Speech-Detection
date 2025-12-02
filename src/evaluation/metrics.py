"""Evaluation metrics wrappers"""
from sklearn.metrics import classification_report, f1_score, accuracy_score


def classification_results(y_true, y_pred):
    return classification_report(y_true, y_pred, digits=4)

def f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')

def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)
