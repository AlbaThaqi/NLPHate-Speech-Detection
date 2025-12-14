from collections import Counter
from typing import List


def pad_sequence(seq, max_len, pad_value=0):
    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + [pad_value] * (max_len - len(seq))

class Tokenizer:
    def __init__(self, min_freq: int = 2, max_vocab_size: int = 30000):
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size

        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}

    def fit(self, texts: List[str]):
        counter = Counter()
        for text in texts:
            counter.update(text.split())

        # Sort by frequency
        most_common = counter.most_common(self.max_vocab_size)

        for word, freq in most_common:
            if freq >= self.min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def text_to_sequence(self, text: str):
        return [
            self.word2idx.get(word, self.word2idx["<UNK>"])
            for word in text.split()
        ]

    def encode(self, texts: List[str]):
        return [self.text_to_sequence(t) for t in texts]

    def vocab_size(self):
        return len(self.word2idx)
