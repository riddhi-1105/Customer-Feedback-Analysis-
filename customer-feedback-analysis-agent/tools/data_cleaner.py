"""
Data Cleaning Tool
==================
Cleans and normalizes raw customer feedback data.
"""

import re
import pandas as pd
from typing import Tuple, Dict


def clean_text(text: str) -> str:
    """Normalize a single feedback string."""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove non-printable chars
    text = re.sub(r"[^\x20-\x7E]", " ", text)
    return text.strip()


def run(df: pd.DataFrame, feedback_col: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Clean the feedback dataframe.

    Parameters
    ----------
    df : pd.DataFrame  — raw uploaded dataframe
    feedback_col : str — name of the feedback text column

    Returns
    -------
    cleaned_df : pd.DataFrame
    quality_report : dict
    """
    report = {
        "records_received": len(df),
        "records_processed": 0,
        "duplicates_removed": 0,
        "missing_handled": 0,
        "short_removed": 0,
        "valid_records": 0,
    }

    # Handle completely empty dataframe
    if len(df) == 0:
        return df, report

    df = df.copy()
    # Ensure column names are unique strings without whitespace
    clean_cols = []
    seen = {}
    for c in df.columns:
        name = str(c).strip()
        if name in seen:
            seen[name] += 1
            clean_cols.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            clean_cols.append(name)
    df.columns = clean_cols

    if feedback_col not in df.columns:
        return df, report

    # Drop rows where feedback column is fully null
    before = len(df)
    df = df.dropna(subset=[feedback_col])
    report["missing_handled"] = before - len(df)

    # Normalize text
    df = df.copy()
    df[feedback_col] = df[feedback_col].apply(clean_text)

    # Remove completely empty after clean (only if there are rows)
    if len(df) > 0:
        try:
            empty_mask = df[feedback_col].astype(str).str.strip() == ""
            report["missing_handled"] += empty_mask.sum()
            df = df[~empty_mask]
        except Exception:
            pass

    # Remove duplicates (exact same feedback text)
    before = len(df)
    df = df.drop_duplicates(subset=[feedback_col], keep="first")
    report["duplicates_removed"] = before - len(df)

    # Remove very short feedback (< 3 words) — likely noise
    before = len(df)
    df = df[df[feedback_col].str.split().str.len() >= 3]
    report["short_removed"] = before - len(df)

    # Reset index
    df = df.reset_index(drop=True)
    df["feedback_id"] = df.index + 1

    report["records_processed"] = len(df)
    report["valid_records"] = len(df)

    return df, report
