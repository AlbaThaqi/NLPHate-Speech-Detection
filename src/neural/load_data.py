"""Data loader for neural models reusing HateXplain format.

Provides `load_hatexplain(path)` -> (DataFrame, label2id) and
`get_class_weights(df)` for imbalance handling.
"""
import json
from pathlib import Path
from collections import Counter
from typing import Tuple, Dict

import pandas as pd


def load_hatexplain(path: str = "data/dataset.json") -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Load HateXplain JSON and return DataFrame and label2id mapping.

    Expected format: { post_id: {"post_tokens": [...], "annotators": [{"label": "..."}, ...], ... }, ... }
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"HateXplain dataset not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for post_key, entry in data.items():
        tokens = entry.get("post_tokens", []) or []
        text = " ".join(tokens).strip()

        annotators = entry.get("annotators", []) or []
        labels = [a.get("label") for a in annotators if a.get("label")]
        if len(labels) == 0:
            continue
        # majority vote
        majority_label = Counter(labels).most_common(1)[0][0]

        rows.append({"id": post_key, "text": text, "label": majority_label})

    df = pd.DataFrame(rows)
    label2id = {lab: idx for idx, lab in enumerate(sorted(df["label"].unique()))}
    df["label_id"] = df["label"].map(label2id)

    return df, label2id


def get_class_weights(df, label_col: str = "label_id") -> Dict[int, float]:
    """Return class weights dict usable in PyTorch training loops or sklearn.
    Uses sklearn's compute_class_weight if available.
    """
    try:
        from sklearn.utils.class_weight import compute_class_weight
        import numpy as np
        classes = np.unique(df[label_col].values)
        weights = compute_class_weight("balanced", classes=classes, y=df[label_col].values)
        return {int(c): float(w) for c, w in zip(classes, weights)}
    except Exception:
        # fallback to uniform weights
        classes = sorted(df[label_col].unique())
        return {int(c): 1.0 for c in classes}
