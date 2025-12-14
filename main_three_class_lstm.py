import torch
import torch.nn as nn
from src.three_class_lstm.tokenizer import Tokenizer
from src.three_class_lstm.dataset import create_dataloaders
from src.three_class_lstm.model import LSTMClassifier
from src.three_class_lstm.train import train_epoch, eval_epoch
from src.three_class_lstm.embeddings import load_glove_embeddings

DATA_PATH = "data/dataset.json"

BATCH_SIZE = 32
MAX_LEN = 50
EPOCHS = 8
LR = 1e-3


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print("Loading data...")
    texts, labels = [], []

    # Load raw texts for tokenizer fitting
    from src.three_class_lstm.dataset import load_hatexplain_json
    texts, _ = load_hatexplain_json(DATA_PATH)

    tokenizer = Tokenizer()
    tokenizer.fit(texts)

    print("Vocab size:", tokenizer.vocab_size())

    glove_path = "../data/glove.6B.200d.txt"
    embeddings = load_glove_embeddings(glove_path, tokenizer)

    train_loader, val_loader, test_loader = create_dataloaders(
        DATA_PATH, tokenizer,
        batch_size=BATCH_SIZE,
        max_len=MAX_LEN
    )

    model = LSTMClassifier(
        vocab_size=tokenizer.vocab_size(),
        embeddings=embeddings,
        freeze_embeddings=True
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_f1 = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc, val_f1 = eval_epoch(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch} | "
            f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} F1 {train_f1:.4f} | "
            f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} F1 {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "models/lstm_best.pt")

    print("Training complete. Best Val F1:", best_val_f1)

    print("Evaluating on test set...")
    model.load_state_dict(torch.load("models/lstm_best.pt"))
    test_loss, test_acc, test_f1 = eval_epoch(
        model, test_loader, criterion, device
    )

    print(f"Test Acc: {test_acc:.4f} | Test Macro-F1: {test_f1:.4f}")


if __name__ == "__main__":
    main()
