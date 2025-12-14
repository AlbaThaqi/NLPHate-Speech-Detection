"""Neural model helpers for hate-speech detection.

This package provides lightweight data loading, training wrappers for LSTM/CNN
and a DistilBERT fine-tune wrapper, plus simple interpretability and an
HTML viewer to save results under analysis/neural.
"""

from .data_loader import load_hatexplain_dataset
from .train_neural import train_lstm, train_cnn, train_transformer, evaluate
from .interpret import explain_text
from .results_viewer import save_results_html
from .results_viewer_advanced import save_results_html_advanced

__all__ = [
    "load_hatexplain_dataset",
    "train_lstm",
    "train_cnn",
    "train_transformer",
    "evaluate",
    "explain_text",
    "save_results_html",
    "save_results_html_advanced",
]
