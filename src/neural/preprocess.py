"""Preprocessing utilities for neural models.

Provides a simple SequenceTokenizer (vocab build + encode + pad) and a
TransformerTokenizer wrapper around Hugging Face `AutoTokenizer`.
"""
from typing import List, Dict
import numpy as np


class SequenceTokenizer:
    """Simple whitespace tokenizer + vocab builder.

    Methods:
    - build_vocab(texts): builds word->index mapping (0 reserved for PAD, 1 for OOV)
    - encode(text): returns list[int]
    - encode_batch(texts): returns np.ndarray padded to max_length
    """

    def __init__(self, vocab_size: int = 10000, max_length: int = 256):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.word2idx = {"<PAD>": 0, "<OOV>": 1}

    def build_vocab(self, texts: List[str]):
        freq = {}
        for t in texts:
            for w in t.split():
                freq[w] = freq.get(w, 0) + 1
        # sort by frequency
        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        for idx, (w, _) in enumerate(sorted_words[: self.vocab_size - 2], start=2):
            self.word2idx[w] = idx

    def encode(self, text: str) -> List[int]:
        ids = [self.word2idx.get(w, 1) for w in text.split()]
        if len(ids) >= self.max_length:
            return ids[: self.max_length]
        # pad
        ids = ids + [0] * (self.max_length - len(ids))
        return ids

    def encode_batch(self, texts: List[str]):
        arr = np.array([self.encode(t) for t in texts], dtype=np.int64)
        return arr


class TransformerTokenizer:
    """Wrapper around Hugging Face tokenizer.

    Returns dict with input_ids, attention_mask as numpy arrays.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased", max_length: int = 128):
        try:
            from transformers import AutoTokenizer
        except Exception as e:
            raise ImportError("Install 'transformers' to use TransformerTokenizer: pip install transformers") from e
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.max_length = max_length

    def tokenize_batch(self, texts: List[str]) -> Dict[str, np.ndarray]:
        enc = self.tokenizer(
            list(texts), padding="max_length", truncation=True, max_length=self.max_length, return_tensors="np"
        )
        return {k: v for k, v in enc.items()}
