import json
import pandas as pd
from pathlib import Path
from collections import Counter


def load_hatexplain(path: str = "../../data/dataset.json"):
    """
    Loads and flattens the HateXplain dataset into a clean pandas DataFrame.

    Expected fields:
        - post_tokens: list of tokens
        - annotators: list of {label: "...", annotator_id: ..}
    """
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for post_key, entry in data.items():
        tokens = entry.get("post_tokens", [])
        text = " ".join(tokens)

        # Extract all labels from annotators
        annotator_labels = [ann["label"] for ann in entry.get("annotators", [])]

        if len(annotator_labels) == 0:
            continue

        # Majority vote
        majority_label = Counter(annotator_labels).most_common(1)[0][0]

        rows.append({
            "id": post_key,
            "text": text,
            "label": majority_label
        })

    return pd.DataFrame(rows)
