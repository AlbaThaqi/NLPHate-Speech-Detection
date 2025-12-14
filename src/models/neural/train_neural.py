import os
from typing import Tuple, Optional, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from .data_loader import TextDataset, load_hatexplain_dataset


class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden=128, num_classes=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):
        emb = self.emb(x)
        out, _ = self.lstm(emb)
        out = out.permute(0, 2, 1)
        pooled = self.pool(out).squeeze(-1)
        return self.fc(pooled)


class SimpleCNN(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, num_classes=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.conv = nn.Conv1d(emb_dim, 128, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        emb = self.emb(x).permute(0, 2, 1)
        c = torch.relu(self.conv(emb))
        p = self.pool(c).squeeze(-1)
        return self.fc(p)


def _train_model(model, train_loader, dev_loader, epochs=3, lr=1e-3, device=None, class_weights=None) -> List[Dict]:
    """Train model and return per-epoch metrics history."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if class_weights is not None:
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))
    else:
        loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []  # track metrics per epoch
    for ep in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        # evaluate after each epoch
        metrics = evaluate(model, dev_loader, device=device)
        metrics["epoch"] = ep + 1
        history.append(metrics)
        print(f"  Epoch {ep+1}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}")
    return history


def evaluate(model, loader, device=None) -> Dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            ys.extend(yb.numpy().tolist())
            ps.extend(preds.tolist())
    acc = accuracy_score(ys, ps)
    f1 = f1_score(ys, ps, average="macro")
    cm = confusion_matrix(ys, ps)
    return {"accuracy": acc, "f1_macro": f1, "confusion_matrix": cm.tolist()}


def train_lstm(train_examples=None, dev_examples=None, epochs=3, batch_size=64, save_path: str = "models/neural/lstm.pt") -> Dict:
    """Train LSTM and return history of metrics per epoch."""
    train, dev = train_examples, dev_examples
    if train is None or dev is None:
        train, dev = load_hatexplain_dataset()
    train_ds = TextDataset(train)
    dev_ds = TextDataset(dev, vocab=train_ds.vocab)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size)
    model = SimpleLSTM(vocab_size=len(train_ds.vocab))
    # class weights to mitigate imbalance
    labels = [0 if e["label"] in ("normal", "none") else 1 for e in train]
    from collections import Counter

    c = Counter(labels)
    total = sum(c.values())
    class_weights = [total / (c.get(i, 1)) for i in (0, 1)]
    history = _train_model(model, train_loader, dev_loader, epochs=epochs, class_weights=class_weights)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    return {"history": history, "final": history[-1] if history else {}}


def train_cnn(train_examples=None, dev_examples=None, epochs=3, batch_size=64, save_path: str = "models/neural/cnn.pt") -> Dict:
    """Train CNN and return history of metrics per epoch."""
    train, dev = train_examples, dev_examples
    if train is None or dev is None:
        train, dev = load_hatexplain_dataset()
    train_ds = TextDataset(train)
    dev_ds = TextDataset(dev, vocab=train_ds.vocab)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size)
    model = SimpleCNN(vocab_size=len(train_ds.vocab))
    labels = [0 if e["label"] in ("normal", "none") else 1 for e in train]
    from collections import Counter

    c = Counter(labels)
    total = sum(c.values())
    class_weights = [total / (c.get(i, 1)) for i in (0, 1)]
    history = _train_model(model, train_loader, dev_loader, epochs=epochs, class_weights=class_weights)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    return {"history": history, "final": history[-1] if history else {}}


def train_transformer(
    train_examples=None,
    dev_examples=None,
    model_name: str = "distilbert-base-uncased",
    epochs=3,
    batch_size=16,
    save_path: str = "models/neural/transformer",
    freeze_encoder: Optional[bool] = None,
    max_length: int = 128,
    subset_size: Optional[int] = None,
) -> Dict:
    """Fine-tune a transformer with per-epoch history.

    On GPU: full fine-tuning with larger max_length by default.
    On CPU: frozen encoder, shorter max_length, optional subset for speed.
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    train, dev = train_examples, dev_examples
    if train is None or dev is None:
        train, dev = load_hatexplain_dataset()

    # prepare texts and labels (binary)
    train_texts = [e["text"] for e in train]
    train_labels = [0 if e["label"] in ("normal", "none") else 1 for e in train]

    # optionally subsample for speed
    if subset_size and subset_size > 0 and subset_size < len(train_texts):
        train_texts = train_texts[:subset_size]
        train_labels = train_labels[:subset_size]

    tok = AutoTokenizer.from_pretrained(model_name)
    enc = tok(train_texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # decide freeze default: freeze on CPU runs, fine-tune on GPU by default
    if freeze_encoder is None:
        freeze_encoder = not torch.cuda.is_available()

    # freeze encoder for fast CPU runs
    if freeze_encoder:
        try:
            for param in model.distilbert.parameters():
                param.requires_grad = False
        except Exception:
            pass

    # create dataset
    ds = torch.utils.data.TensorDataset(enc["input_ids"], enc.get("attention_mask"), torch.tensor(train_labels))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    # use torch.optim.AdamW for optimizer (works with transformers)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)
    
    history = []
    for ep in range(epochs):
        model.train()
        for input_ids, attn, yb in loader:
            input_ids = input_ids.to(device)
            attn = attn.to(device)
            yb = yb.to(device)
            out = model(input_ids=input_ids, attention_mask=attn, labels=yb)
            loss = out.loss
            opt.zero_grad()
            loss.backward()
            opt.step()

        # evaluate after each epoch
        dev_texts = [e["text"] for e in dev]
        dev_labels = [0 if e["label"] in ("normal", "none") else 1 for e in dev]
        tok_dev = tok(dev_texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        model.eval()
        with torch.no_grad():
            input_ids = tok_dev["input_ids"].to(device)
            attn = tok_dev["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attn).logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
        acc = accuracy_score(dev_labels, preds)
        f1 = f1_score(dev_labels, preds, average="macro")
        cm = confusion_matrix(dev_labels, preds)
        metrics = {"accuracy": acc, "f1_macro": f1, "confusion_matrix": cm.tolist(), "epoch": ep + 1}
        history.append(metrics)
        print(f"  Epoch {ep+1}: Acc={acc:.4f}, F1={f1:.4f}")

    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tok.save_pretrained(save_path)

    return {"history": history, "final": history[-1] if history else {}}
