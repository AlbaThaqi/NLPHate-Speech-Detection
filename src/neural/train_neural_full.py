"""Entrypoint: full neural training pipeline (LSTM, CNN, optional Transformer).

Run with: python -m src.neural.train_neural_full
"""

# python -c "from src.neural.train_neural_full import main; main(sample_size=50)"
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.model_selection import train_test_split

from src.neural.load_data import load_hatexplain
from src.neural.preprocess import SequenceTokenizer, TransformerTokenizer
from src.neural.train_neural import train_lstm_model, train_cnn_model, TransformerClassifier

logger = logging.getLogger("train_neural_full")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

OUTPUT_DIR = Path("models/neural")
ANALYSIS_DIR = Path("analysis/neural")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


def load_and_split(data_path: str = "data/dataset.json", sample_size: Optional[int] = None, test_size: float = 0.1):
    df, label2id = load_hatexplain(data_path)
    id2label = {v: k for k, v in label2id.items()}
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    texts = df["text"].tolist()
    labels = df["label_id"].values
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=test_size, random_state=42, stratify=labels)
    logger.info(f"Loaded {len(df)} samples. Train: {len(X_train)}, Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test, label2id, id2label


def main(
    data_path: str = "data/dataset.json",
    sample_size: Optional[int] = 500,
    vocab_size: int = 5000,
    max_length: int = 256,
    epochs: int = 3,
    batch_size: int = 16,
    train_transformer: bool = False,
    transformer_model: str = "distilbert-base-uncased",
):
    np.random.seed(42)
    X_train, X_test, y_train, y_test, label2id, id2label = load_and_split(data_path=data_path, sample_size=sample_size)

    # Sequence tokenizer for LSTM/CNN
    seq_tok = SequenceTokenizer(vocab_size=vocab_size, max_length=max_length)
    seq_tok.build_vocab(X_train)
    X_train_enc = seq_tok.encode_batch(X_train)
    X_test_enc = seq_tok.encode_batch(X_test)

    # Split a small validation from train for quick eval
    from sklearn.model_selection import train_test_split
    X_train_enc_small, X_val_enc, y_train_small, y_val = train_test_split(X_train_enc, y_train, test_size=0.1, random_state=42)

    # Train LSTM
    try:
        model_lstm, res_lstm = train_lstm_model(X_train_enc_small, y_train_small, X_val_enc, y_val, num_classes=len(label2id), vocab_size=vocab_size, epochs=epochs, batch_size=batch_size)
        logger.info(f"LSTM Validation -> acc: {res_lstm['accuracy']:.4f}, f1: {res_lstm['f1']:.4f}")
        # save weights if torch is available
        try:
            import torch
            torch.save(model_lstm.state_dict(), OUTPUT_DIR / "lstm.pt")
        except Exception:
            logger.info("torch not available to save LSTM model weights; install torch to save/load models")
    except Exception as e:
        logger.error(f"LSTM training failed: {e}")

    # Train CNN
    try:
        model_cnn, res_cnn = train_cnn_model(X_train_enc_small, y_train_small, X_val_enc, y_val, num_classes=len(label2id), vocab_size=vocab_size, epochs=epochs, batch_size=batch_size)
        logger.info(f"CNN Validation -> acc: {res_cnn['accuracy']:.4f}, f1: {res_cnn['f1']:.4f}")
        try:
            import torch
            torch.save(model_cnn.state_dict(), OUTPUT_DIR / "cnn.pt")
        except Exception:
            logger.info("torch not available to save CNN model weights; install torch to save/load models")
    except Exception as e:
        logger.error(f"CNN training failed: {e}")

    # Optional Transformer
    if train_transformer:
        try:
            # Use TransformerTokenizer to create encodings
            tr_tok = TransformerTokenizer(model_name=transformer_model, max_length=max_length)
            train_enc = tr_tok.tokenize_batch(X_train)
            test_enc = tr_tok.tokenize_batch(X_test)
            classifier = TransformerClassifier(model_name=transformer_model, num_labels=len(label2id))
            classifier.train(train_enc, y_train, epochs=2, batch_size=8)
            preds = classifier.predict(test_enc)
            from sklearn.metrics import accuracy_score, f1_score
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average='macro')
            logger.info(f"Transformer Test -> acc: {acc:.4f}, f1: {f1:.4f}")
        except Exception as e:
            logger.error(f"Transformer training failed: {e}")

    logger.info("Neural training done. Check models/ and analysis/")


if __name__ == '__main__':
    # demo run
    main()
