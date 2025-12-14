# src/train_classical.py
import os
import joblib
import warnings
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
from scipy.sparse import hstack

# Local imports - assumes these exist from earlier steps
from src.load_data import load_hatexplain
from src.preprocess import TextPreprocessor

warnings.filterwarnings("ignore")
np.random.seed(42)


OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR = Path("analysis")
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


def prepare_data(df: pd.DataFrame, test_size=0.2) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    # If labels are strings, keep them but also provide numeric encoding for models that need it
    df = df.copy()
    # Map labels to integers
    label2id = {lab: idx for idx, lab in enumerate(sorted(df["label"].unique()))}
    id2label = {v: k for k, v in label2id.items()}
    df["label_id"] = df["label"].map(label2id)

    X = df["text"]
    y = df["label_id"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    return X_train.reset_index(drop=True), X_test.reset_index(drop=True), y_train.reset_index(drop=True), y_test.reset_index(drop=True), label2id, id2label


def build_vectorizers(max_features_word=50000, max_features_char=30000):
    """
    Return two TF-IDF vectorizers: word-level and char-level.
    """
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=max_features_word,
        strip_accents="unicode",
        norm="l2",
        min_df=3,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        max_features=max_features_char,
        strip_accents="unicode",
        norm="l2",
        min_df=3,
    )
    return word_vectorizer, char_vectorizer


def vectorize_fit_transform(word_vec, char_vec, X_train: pd.Series, X_test: pd.Series):
    """
    Fit vectorizers on X_train, transform both train and test, and horizontally stack them.
    Returns X_train_vec, X_test_vec and the fitted vectorizers.
    """
    X_train_word = word_vec.fit_transform(X_train)
    X_train_char = char_vec.fit_transform(X_train)

    X_test_word = word_vec.transform(X_test)
    X_test_char = char_vec.transform(X_test)

    Xtr = hstack([X_train_word, X_train_char], format="csr")
    Xte = hstack([X_test_word, X_test_char], format="csr")

    return Xtr, Xte, word_vec, char_vec


def train_and_evaluate(Xtr, Xte, y_train, y_test, id2label: Dict[int, str]):
    """
    Train three models and evaluate them. Save models and print results.
    """
    results = {}

    # 1) Logistic Regression (with simple CV on C)
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(solver="saga", max_iter=2000, class_weight="balanced", random_state=42)
    param_grid = {"C": [0.01, 0.1, 1.0, 5.0]}
    grid_lr = GridSearchCV(lr, param_grid, cv=3, scoring="f1_macro", n_jobs=-1, verbose=1)
    grid_lr.fit(Xtr, y_train)
    best_lr = grid_lr.best_estimator_
    print("LR best params:", grid_lr.best_params_)
    y_pred_lr = best_lr.predict(Xte)
    _report_and_save(best_lr, "logreg", y_test, y_pred_lr, id2label)
    results["logreg"] = {"model": best_lr, "y_pred": y_pred_lr}

    # 2) Linear SVM (LinearSVC)
    print("\nTraining LinearSVC...")
    svc = LinearSVC(class_weight="balanced", max_iter=5000, random_state=42)
    param_grid_svc = {"C": [0.01, 0.1, 1.0, 5.0]}
    grid_svc = GridSearchCV(svc, param_grid_svc, cv=3, scoring="f1_macro", n_jobs=-1, verbose=1)
    grid_svc.fit(Xtr, y_train)
    best_svc = grid_svc.best_estimator_
    print("SVC best params:", grid_svc.best_params_)
    y_pred_svc = best_svc.predict(Xte)
    _report_and_save(best_svc, "svc", y_test, y_pred_svc, id2label)
    results["svc"] = {"model": best_svc, "y_pred": y_pred_svc}

    # 3) Multinomial Naive Bayes (fast baseline)
    print("\nTraining MultinomialNB...")
    mnb = MultinomialNB()
    mnb.fit(Xtr, y_train)
    y_pred_mnb = mnb.predict(Xte)
    _report_and_save(mnb, "mnb", y_test, y_pred_mnb, id2label)
    results["mnb"] = {"model": mnb, "y_pred": y_pred_mnb}

    return results


def _report_and_save(model, name: str, y_true, y_pred, id2label):
    """
    Print metrics, confusion matrix, save model to disk.
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"\n{name} Accuracy: {acc:.4f}  Macro-F1: {f1:.4f}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=[id2label[i] for i in sorted(id2label.keys())], digits=4))

    # Save model
    model_path = OUTPUT_DIR / f"{name}.joblib"
    joblib.dump(model, model_path)
    print(f"Saved {name} model to {model_path}")

    # Confusion matrix plot
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(f"{name} Confusion Matrix")
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ticks = np.arange(len(cm))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    # annotate
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.tight_layout()
    fig_path = ANALYSIS_DIR / f"{name}_confusion.png"
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Saved confusion matrix to {fig_path}")


def save_vectorizers(word_vec, char_vec):
    joblib.dump(word_vec, OUTPUT_DIR / "tfidf_word.joblib")
    joblib.dump(char_vec, OUTPUT_DIR / "tfidf_char.joblib")
    print("Saved vectorizers to models/")


def export_misclassifications(X_test: pd.Series, y_test, preds: Dict[str, np.ndarray], id2label):
    """
    Create CSVs of misclassified examples for each model.
    """
    df = pd.DataFrame({"text": X_test, "true_id": y_test})
    df["true_label"] = df["true_id"].map(id2label)
    for name, arr in preds.items():
        df[f"pred_{name}"] = arr
        df[f"pred_label_{name}"] = df[f"pred_{name}"].map(id2label)

        mis = df[df["true_id"] != df[f"pred_{name}"]]
        mis_path = ANALYSIS_DIR / f"misclassified_{name}.csv"
        mis.to_csv(mis_path, index=False)
        print(f"Saved {len(mis)} misclassified examples for {name} to {mis_path}")


def show_top_features(word_vec, char_vec, model, top_n=25):
    """
    Print top positive/negative features for linear models.
    Assumes concatenation of word_vec and char_vec in that order.
    """
    try:
        feature_names_word = word_vec.get_feature_names_out()
        feature_names_char = char_vec.get_feature_names_out()
    except Exception:
        feature_names_word = word_vec.get_feature_names()
        feature_names_char = char_vec.get_feature_names()

    all_features = np.concatenate([feature_names_word, feature_names_char])

    if hasattr(model, "coef_"):
        # multiclass: coef_ shape (n_classes, n_features) for LogisticRegression; for LinearSVC same
        coefs = model.coef_
        # For multiclass, print top features per class
        for cls_idx in range(coefs.shape[0]):
            coef = coefs[cls_idx]
            top_pos = np.argsort(coef)[-top_n:][::-1]
            top_neg = np.argsort(coef)[:top_n]
            print(f"\nClass {cls_idx} top + features:")
            print(", ".join(all_features[top_pos]))
            print(f"\nClass {cls_idx} top - features:")
            print(", ".join(all_features[top_neg]))
    elif isinstance(model, MultinomialNB):
        # For NB, use feature_log_prob_
        flp = model.feature_log_prob_
        for cls_idx in range(flp.shape[0]):
            top = np.argsort(flp[cls_idx])[-top_n:][::-1]
            print(f"\nNB Class {cls_idx} top features:")
            print(", ".join(all_features[top]))
    else:
        print("Top feature extraction not supported for this model type.")


def try_lime_explain(pipeline_predict_proba, X_text_sample: list, class_names: list):
    try:
        from lime.lime_text import LimeTextExplainer

        explainer = LimeTextExplainer(class_names=class_names)
        exp = explainer.explain_instance(X_text_sample[0], pipeline_predict_proba, num_features=10)
        print("\nLIME explanation (as list of features -> weight):")
        print(exp.as_list())
    except Exception as exc:
        print("LIME not available or errored:", exc)


def main():
    print("Loading dataset...")
    df = load_hatexplain("../data/dataset.json")  # adjust path if different
    print("Total records:", len(df))
    X_train_raw, X_test_raw, y_train, y_test, label2id, id2label = prepare_data(df)

    # Preprocess (lemmatize + stopwords removed) - we use the scikit transformer
    pre = TextPreprocessor(do_lemmatize=True)
    print("Preprocessing train set...")
    X_train = pre.transform(X_train_raw)
    print("Preprocessing test set...")
    X_test = pre.transform(X_test_raw)

    # Vectorizers
    print("Building TF-IDF vectorizers (word + char)...")
    word_vec, char_vec = build_vectorizers()
    Xtr, Xte, word_vec, char_vec = vectorize_fit_transform(word_vec, char_vec, X_train, X_test)

    print("Saving vectorizers...")
    save_vectorizers(word_vec, char_vec)

    # Train & evaluate models
    results = train_and_evaluate(Xtr, Xte, y_train, y_test, id2label)

    # Export misclassifications
    preds = {name: results[name]["y_pred"] for name in results}
    export_misclassifications(X_test_raw, y_test, preds, id2label)

    # Show top features for best models
    print("\nTop features for Logistic Regression:")
    show_top_features(word_vec, char_vec, results["logreg"]["model"], top_n=25)

    print("\nTop features for LinearSVC:")
    show_top_features(word_vec, char_vec, results["svc"]["model"], top_n=25)

    print("\nTop features for MultinomialNB:")
    show_top_features(word_vec, char_vec, results["mnb"]["model"], top_n=25)

    try:
        # Create a predict_proba wrapper if the model supports it (LogisticRegression does)
        def pipeline_predict_proba(texts):
            Xw = word_vec.transform(texts)
            Xc = char_vec.transform(texts)
            Xboth = hstack([Xw, Xc], format="csr")
            # Use logistic regression's predict_proba if available, else use svc decision_function as a proxy
            if hasattr(results["logreg"]["model"], "predict_proba"):
                return results["logreg"]["model"].predict_proba(Xboth)
            else:
                # convert decision function to pseudo-probabilities
                d = results["svc"]["model"].decision_function(Xboth)
                # simple softmax
                expd = np.exp(d - np.max(d, axis=1, keepdims=True))
                return expd / expd.sum(axis=1, keepdims=True)

        sample_text = [X_test_raw.iloc[0]]
        try_lime_explain(pipeline_predict_proba, sample_text, [id2label[i] for i in sorted(id2label.keys())])
    except Exception as exc:
        print("LIME explanation skipped:", exc)


if __name__ == "__main__":
    main()
