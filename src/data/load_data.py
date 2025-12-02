"""Simple data loader utilities

Usage:
    from src.data.load_data import load_csv
    df = load_csv('data/raw/my_dataset.csv')
"""
import pandas as pd
from typing import Optional


def load_csv(path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame.

    Args:
        path: path to CSV file
        nrows: optional number of rows to read (for quick tests)

    Returns:
        DataFrame
    """
    df = pd.read_csv(path, nrows=nrows)
    return df
