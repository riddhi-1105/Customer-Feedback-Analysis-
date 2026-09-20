"""
Trend Analyzer Tool
===================
Analyzes trends over time if a date column is present.
Returns time-series data for sentiment, volume, topics, and issues.
"""

import pandas as pd
from typing import Dict, Optional, List


def run(df: pd.DataFrame, date_col: Optional[str] = None) -> Dict:
    """
    Compute trend data from the dataframe.

    Returns a dict with trend DataFrames or None if no date available.
    """
    result = {
        "date_available": False,
        "sentiment_trend": None,
        "volume_trend": None,
        "topic_trend": None,
        "issue_trend": None,
    }

    if not date_col or date_col not in df.columns:
        return result

    # Parse dates
    try:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        if len(df) == 0:
            return result
    except Exception:
        return result

    result["date_available"] = True
    df["_period"] = df[date_col].dt.to_period("M").astype(str)

    # 1. Volume Trend — feedback count per month
    volume_trend = df.groupby("_period").size().reset_index(name="count")
    volume_trend.columns = ["period", "count"]
    result["volume_trend"] = volume_trend

    # 2. Sentiment Trend — sentiment counts per month
    if "sentiment" in df.columns:
        sentiment_trend = (
            df.groupby(["_period", "sentiment"])
            .size()
            .reset_index(name="count")
            .rename(columns={"_period": "period"})
        )
        result["sentiment_trend"] = sentiment_trend

    # 3. Topic Trend — top 5 topics per month
    if "topic" in df.columns:
        topic_trend = (
            df.groupby(["_period", "topic"])
            .size()
            .reset_index(name="count")
            .rename(columns={"_period": "period"})
        )
        result["topic_trend"] = topic_trend

    # 4. Issue Trend — complaint volume per month
    if "detected_issue" in df.columns:
        issue_trend = (
            df[df["detected_issue"] != "General Feedback"]
            .groupby("_period")
            .size()
            .reset_index(name="complaints")
            .rename(columns={"_period": "period"})
        )
        result["issue_trend"] = issue_trend

    return result
