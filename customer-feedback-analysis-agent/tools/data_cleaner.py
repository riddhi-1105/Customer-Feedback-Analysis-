"""
Data Cleaning Tool
==================
Cleans and normalizes raw customer feedback data.

Design notes
------------
* Unicode is preserved (Hindi/Marathi words, emojis, the rupee sign ...).
  Only control characters are stripped.
* De-duplication is done on the *whole record* (feedback + customer + date +
  product ...), NOT on the text alone. Real customers frequently write the
  same short sentence ("Great product!"); deleting those would silently throw
  away valid, distinct feedback.
"""

import re
from typing import Dict, Tuple

import pandas as pd

MIN_WORDS = 2  # anything shorter is noise ("ok", "bad")

_ID_LIKE = re.compile(r"(^id$|_id$|^index$|^unnamed)", re.IGNORECASE)
# ids that identify a *customer* (kept for de-duplication) vs a *row number*
_CUSTOMER_ID = re.compile(r"(customer|user|client|account)", re.IGNORECASE)


def clean_text(text) -> str:
    """Normalize a single feedback string."""
    if not isinstance(text, str):
        return ""
    # strip control characters but keep all printable unicode
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _dedupe_subset(df: pd.DataFrame, feedback_col: str) -> list:
    """Columns that define a duplicate record (excludes row-number style ids)."""
    subset = []
    for c in df.columns:
        if c == feedback_col:
            subset.append(c)
        elif _ID_LIKE.search(str(c)) and not _CUSTOMER_ID.search(str(c)):
            continue  # feedback_id / index: unique per row, would hide duplicates
        else:
            subset.append(c)
    return subset or [feedback_col]


def run(df: pd.DataFrame, feedback_col: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Clean the feedback dataframe.

    Returns (cleaned_df, quality_report)
    """
    report = {
        "records_received": len(df),
        "records_processed": 0,
        "duplicates_removed": 0,
        "missing_handled": 0,
        "short_removed": 0,
        "valid_records": 0,
    }

    if len(df) == 0:
        return df, report

    df = df.copy()

    # unique, whitespace-free string column names
    clean_cols, seen = [], {}
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

    # missing
    before = len(df)
    df = df.dropna(subset=[feedback_col])
    report["missing_handled"] = before - len(df)

    # normalise text
    df[feedback_col] = df[feedback_col].astype(str).apply(clean_text)

    if len(df) > 0:
        empty = df[feedback_col].str.strip() == ""
        report["missing_handled"] += int(empty.sum())
        df = df[~empty]

    # duplicates = identical *records*
    before = len(df)
    df = df.drop_duplicates(subset=_dedupe_subset(df, feedback_col), keep="first")
    report["duplicates_removed"] = before - len(df)

    # too short
    before = len(df)
    df = df[df[feedback_col].str.split().str.len() >= MIN_WORDS]
    report["short_removed"] = before - len(df)

    df = df.reset_index(drop=True)
    # keep a supplied feedback_id if it is unique, otherwise create one
    if "feedback_id" not in df.columns or df["feedback_id"].duplicated().any():
        df["feedback_id"] = df.index + 1

    report["records_processed"] = len(df)
    report["valid_records"] = len(df)
    # plain python ints (numpy ints break json / streamlit display)
    return df, {k: int(v) for k, v in report.items()}
