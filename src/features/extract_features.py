"""Feature extraction helpers (stubs)

Example functions for vectorizing text with scikit-learn.
"""
from typing import Tuple, List
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def vectorize_texts(texts: List[str], method: str = "tfidf", max_features: int = 20000) -> Tuple[object, object]:
    """Return (vectorizer, X).

    method: 'tfidf' or 'count'
    """
    if method == "tfidf":
        vec = TfidfVectorizer(max_features=max_features)
    else:
        vec = CountVectorizer(max_features=max_features)
    X = vec.fit_transform(texts)
    return vec, X
