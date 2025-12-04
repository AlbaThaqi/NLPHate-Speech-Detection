"""Data loading and simple preprocessing utilities.

This module provides safe CSV loading, basic text cleaning and a reproducible
train/val/test splitting helper. The functions are intentionally small and
dependency-light so they can be used in notebooks and scripts.
"""
from pathlib import Path
import re
from typing import Optional, Tuple, Dict

import pandas as pd
from sklearn.model_selection import train_test_split


def load_csv(path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """Safely load a CSV into a pandas DataFrame.

    Args:
        path: file path to CSV
        nrows: optional number of rows to read for quick tests

    Returns:
        DataFrame
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    try:
        df = pd.read_csv(path, nrows=nrows)
    except Exception:
        # try with engine fallback
        df = pd.read_csv(path, engine="python", nrows=nrows)
    return df


def clean_text(text: str) -> str:
    """Basic text cleaning for social media style text.

    Removes URLs, mentions and excessive whitespace and lowercases the text.
    This is intentionally simple — more advanced tokenization / normalization
    should be added to `src/features/` when needed.
    """
    if not isinstance(text, str):
        return ""
    text = text.strip()
    # remove URLs
    text = re.sub(r"http\S+|www\.[^\s]+", "", text)
    # remove mentions
    text = re.sub(r"@\w+", "", text)
    # remove hashtags marker but keep the word
    text = re.sub(r"#", "", text)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text)
    # lowercase
    text = text.lower()
    return text


def preprocess_df(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: Optional[str] = "label",
) -> pd.DataFrame:
    """Apply basic cleaning to a DataFrame and return a cleaned copy.

    Drops rows with empty text after cleaning. Does not modify original df.
    """
    df = df.copy()
    if text_col not in df.columns:
        raise KeyError(f"Text column '{text_col}' not found in DataFrame")
    # ensure string type
    df[text_col] = df[text_col].astype(str).fillna("")
    df["_clean_text"] = df[text_col].map(clean_text)
    # drop empty texts
    df = df[df["_clean_text"].str.strip() != ""]
    # keep cleaned text under the same column name for downstream use
    df[text_col] = df.pop("_clean_text")
    if label_col and label_col in df.columns:
        df[label_col] = df[label_col].astype(str)
    return df


def split_and_save(
    df: pd.DataFrame,
    out_dir: str = "data/processed",
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    text_col: str = "text",
    label_col: Optional[str] = "label",
) -> Dict[str, pd.DataFrame]:
    """Split dataset into train/val/test and save CSVs to out_dir.

    Returns a dict with DataFrames for 'train', 'val', 'test'. If `label_col`
    is present the split is stratified on labels when possible.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if sum(ratios) <= 0:
        raise ValueError("Ratios must sum to a positive number")
    train_ratio, val_ratio, test_ratio = ratios
    # normalize ratios
    total = train_ratio + val_ratio + test_ratio
    train_ratio /= total
    val_ratio /= total
    test_ratio /= total

    if label_col and label_col in df.columns:
        stratify = df[label_col]
        # if any class has fewer than 2 samples, stratified split will fail
        # in that case fall back to non-stratified split
        vc = stratify.value_counts()
        if (vc < 2).any():
            stratify = None
    else:
        stratify = None

    # first split train vs temp (val+test)
    df_train, df_temp = train_test_split(
        df, test_size=(1 - train_ratio), random_state=seed, stratify=stratify
    )

    # compute val proportion relative to temp
    if val_ratio + test_ratio == 0:
        df_val = pd.DataFrame(columns=df.columns)
        df_test = pd.DataFrame(columns=df.columns)
    else:
        val_relative = val_ratio / (val_ratio + test_ratio)
        if stratify is not None and label_col in df_temp.columns:
            stratify_temp = df_temp[label_col]
        else:
            stratify_temp = None
        df_val, df_test = train_test_split(
            df_temp, test_size=(1 - val_relative), random_state=seed, stratify=stratify_temp
        )

    # Save files
    train_fp = out_path / "train.csv"
    val_fp = out_path / "val.csv"
    test_fp = out_path / "test.csv"
    df_train.to_csv(train_fp, index=False)
    df_val.to_csv(val_fp, index=False)
    df_test.to_csv(test_fp, index=False)

    return {"train": df_train, "val": df_val, "test": df_test}

