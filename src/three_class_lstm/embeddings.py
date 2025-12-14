import numpy as np
import torch


def load_glove_embeddings(glove_path, tokenizer, embed_dim=200):
    embeddings = np.random.normal(
        scale=0.6,
        size=(tokenizer.vocab_size(), embed_dim)
    )

    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.rstrip().split(" ")
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")

            if word in tokenizer.word2idx:
                embeddings[tokenizer.word2idx[word]] = vector

    return torch.tensor(embeddings, dtype=torch.float)
