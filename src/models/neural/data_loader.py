import json
import os
from typing import List, Tuple, Dict, Optional

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


def _read_dataset(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _normalize_example(k, entry):
    # dataset.json contains tokenized posts in `post_tokens` and annotators
    tokens = entry.get("post_tokens") or entry.get("tokens")
    text = " ".join(tokens) if tokens else entry.get("post_text") or entry.get("text") or ""
    # majority vote label from annotators
    ann = entry.get("annotators", [])
    if ann:
        labels = [a.get("label") for a in ann if a.get("label")]
        # simple majority vote
        from collections import Counter

        label = Counter(labels).most_common(1)[0][0]
    else:
        label = entry.get("label") or "normal"
    # rationales if present (list of token-level binary masks)
    rationales = entry.get("rationales", [])
    return {"id": k, "text": text, "label": label, "rationales": rationales}


def load_hatexplain_dataset(dataset_path: str = None, test_size: float = 0.2, random_state: int = 42) -> Tuple[List[dict], List[dict]]:
    """Load dataset.json (HateXplain-style) and return train/test lists of examples.

    Each example is a dict {id, text, label, rationales}.
    """
    if dataset_path is None:
        dataset_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "dataset.json")
        dataset_path = os.path.abspath(dataset_path)
    data = _read_dataset(dataset_path)
    examples = [_normalize_example(k, v) for k, v in data.items()]
    labels = [e["label"] for e in examples]
    train, test = train_test_split(examples, test_size=test_size, stratify=labels, random_state=random_state)
    return train, test


class TextDataset(Dataset):
    def __init__(self, examples: List[dict], vocab: Optional[Dict[str, int]] = None, max_len: int = 128):
        self.examples = examples
        self.max_len = max_len
        if vocab is None:
            self.vocab = {"<PAD>": 0, "<UNK>": 1}
            for ex in examples:
                for tok in ex["text"].split():
                    if tok not in self.vocab:
                        self.vocab[tok] = len(self.vocab)
        else:
            self.vocab = vocab

    def _encode(self, text: str):
        ids = [self.vocab.get(t, self.vocab["<UNK>"]) for t in text.split()][: self.max_len]
        pad_len = max(0, self.max_len - len(ids))
        ids = ids + [self.vocab["<PAD>"]] * pad_len
        return torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        x = self._encode(ex["text"])
        y = 0 if ex["label"] in ("normal", "none", "normal") else 1
        return x, torch.tensor(y, dtype=torch.long)
