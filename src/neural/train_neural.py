"""Neural model training helpers (LSTM, CNN, Transformer wrapper).

NOTE: These functions use PyTorch. If `torch` is not installed a clear
ImportError will be raised advising how to install it.
"""
from typing import Tuple, Dict

# Delay torch import until functions are called so module import doesn't fail

def _ensure_torch():
    try:
        import torch
        import torch.nn as nn
    except Exception as e:
        raise ImportError("PyTorch is required to run neural training. Install via: pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118") from e
    return torch, nn


def train_lstm_model(
    X_train, y_train, X_val, y_val,
    num_classes: int = 3,
    vocab_size: int = 10000,
    embedding_dim: int = 100,
    hidden_dim: int = 128,
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    label_names: dict = None,
) -> Tuple[object, Dict]:
    """Train a simple PyTorch LSTM classifier.

    Returns (model, results_dict) where results_dict contains accuracy and f1 on validation.
    This is a minimal implementation intended as a starting point; users should
    adapt training loops and device placement for production.
    """
    torch, nn = _ensure_torch()
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score

    # Minimal dataset wrapping
    X_train = torch.LongTensor(X_train)
    X_val = torch.LongTensor(X_val)
    y_train = torch.LongTensor(y_train).long()
    y_val = torch.LongTensor(y_val).long()

    class SimpleLSTM(nn.Module):
        def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(hidden_dim * 2, num_classes)

        def forward(self, x):
            x = self.emb(x)
            out, _ = self.lstm(x)
            # mean pooling over time
            out = out.mean(dim=1)
            return self.fc(out)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleLSTM(vocab_size, embedding_dim, hidden_dim, num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    # tiny training loop
    model.train()
    for ep in range(epochs):
        permutation = torch.randperm(X_train.size(0))
        for i in range(0, X_train.size(0), batch_size):
            idx = permutation[i : i + batch_size]
            batch_x = X_train[idx].to(device)
            batch_y = y_train[idx].to(device)
            opt.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            opt.step()

    # eval on val
    model.eval()
    with torch.no_grad():
        preds = model(X_val.to(device)).argmax(dim=1).cpu().numpy()
    acc = accuracy_score(y_val.numpy(), preds)
    f1 = f1_score(y_val.numpy(), preds, average="macro")
    results = {"accuracy": float(acc), "f1": float(f1)}
    return model, results


def train_cnn_model(
    X_train, y_train, X_val, y_val,
    num_classes: int = 3,
    vocab_size: int = 10000,
    embedding_dim: int = 100,
    num_filters: int = 100,
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    label_names: dict = None,
) -> Tuple[object, Dict]:
    """Train a simple CNN text classifier (Kim-style).
    """
    torch, nn = _ensure_torch()
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score

    X_train = torch.LongTensor(X_train)
    X_val = torch.LongTensor(X_val)
    y_train = torch.LongTensor(y_train).long()
    y_val = torch.LongTensor(y_val).long()

    class TextCNN(nn.Module):
        def __init__(self, vocab_size, embed_dim, num_filters, num_classes, kernel_sizes=(3,4,5)):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.convs = nn.ModuleList([nn.Conv2d(1, num_filters, (k, embed_dim)) for k in kernel_sizes])
            self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)

        def forward(self, x):
            x = self.emb(x)  # (B, L, E)
            x = x.unsqueeze(1)  # (B, 1, L, E)
            convs = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]
            pools = [torch.max(c, dim=2)[0] for c in convs]
            cat = torch.cat(pools, dim=1)
            return self.fc(cat)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextCNN(vocab_size, embedding_dim, num_filters, num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for ep in range(epochs):
        permutation = torch.randperm(X_train.size(0))
        for i in range(0, X_train.size(0), batch_size):
            idx = permutation[i : i + batch_size]
            batch_x = X_train[idx].to(device)
            batch_y = y_train[idx].to(device)
            opt.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_val.to(device)).argmax(dim=1).cpu().numpy()
    acc = accuracy_score(y_val.numpy(), preds)
    f1 = f1_score(y_val.numpy(), preds, average="macro")
    results = {"accuracy": float(acc), "f1": float(f1)}
    return model, results


class TransformerClassifier:
    """Light wrapper around Hugging Face transformers for training/prediction.

    This requires `transformers` and `torch` installed. Training uses the
    built-in Trainer API if available; for now this wrapper implements a
    minimal fine-tuning interface using Trainer if present.
    """
    def __init__(self, model_name: str = "distilbert-base-uncased", num_labels: int = 3):
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
        except Exception as e:
            raise ImportError("Install 'transformers' and 'torch' to use TransformerClassifier: pip install transformers torch") from e
        import torch
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def train(self, encodings, labels, epochs: int = 3, batch_size: int = 8, learning_rate: float = 2e-5):
        # Simple Trainer-based training
        from transformers import Trainer, TrainingArguments
        import torch
        import numpy as np

        class Dataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
                item["labels"] = torch.tensor(self.labels[idx])
                return item

        train_dataset = Dataset(encodings, labels)

        training_args = TrainingArguments(
            output_dir="./models/transformer", num_train_epochs=epochs, per_device_train_batch_size=batch_size,
            learning_rate=learning_rate, logging_steps=50, save_strategy="no"
        )

        trainer = Trainer(
            model=self.model, args=training_args, train_dataset=train_dataset
        )
        trainer.train()

    def predict(self, encodings):
        import torch
        self.model.eval()
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(encodings["input_ids"]), torch.tensor(encodings["attention_mask"])
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=32)
        preds = []
        with torch.no_grad():
            for batch in loader:
                input_ids, attention_mask = [b.to(self.model.device) for b in batch]
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.argmax(dim=1).cpu().numpy()
                preds.extend(list(logits))
        return preds
