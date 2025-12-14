import json
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from .preprocess import clean_text
from .tokenizer import pad_sequence


LABEL2ID = {
    "normal": 0,
    "offensive": 1,
    "hatespeech": 2
}


class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=50):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = clean_text(self.texts[idx])
        seq = self.tokenizer.text_to_sequence(text)
        seq = pad_sequence(seq, self.max_len)

        return {
            "input_ids": torch.tensor(seq, dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }


def load_hatexplain_json(path):
    texts = []
    labels = []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for _, entry in data.items():
        text = " ".join(entry.get("post_tokens", []))
        annotators = entry.get("annotators", [])

        if not annotators:
            continue

        label = annotators[0].get("label")

        if label not in LABEL2ID:
            continue

        texts.append(text)
        labels.append(LABEL2ID[label])

    return texts, labels


def create_dataloaders(
    data_path,
    tokenizer,
    batch_size=32,
    max_len=50,
    test_size=0.2,
    val_size=0.1
):
    texts, labels = load_hatexplain_json(data_path)

    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=test_size + val_size,
        stratify=labels, random_state=42
    )

    val_ratio = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - val_ratio,
        stratify=y_temp, random_state=42
    )

    train_ds = HateSpeechDataset(X_train, y_train, tokenizer, max_len)
    val_ds = HateSpeechDataset(X_val, y_val, tokenizer, max_len)
    test_ds = HateSpeechDataset(X_test, y_test, tokenizer, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    return train_loader, val_loader, test_loader
