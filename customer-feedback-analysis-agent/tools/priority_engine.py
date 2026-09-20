"""
Priority Engine
===============
Calculates priority score for each feedback / issue cluster.
Factors: sentiment, emotion severity, complaint flag, issue frequency,
         business impact keywords, urgency signals.

Output: HIGH / MEDIUM / LOW with explanation.
"""

import pandas as pd
from typing import Dict, List, Tuple


# ── Weight constants ─────────────────────────────────────────────────────────
SENTIMENT_WEIGHT = {
    "Negative": 3,
    "Neutral": 1,
    "Positive": 0,
}

EMOTION_WEIGHT = {
    "Angry": 4,
    "Frustrated": 3,
    "Disappointed": 2,
    "Confused": 1,
    "Neutral": 0,
    "Satisfied": 0,
    "Happy": 0,
}

URGENCY_KEYWORDS = [
    "urgent", "immediately", "asap", "critical", "emergency", "broken",
    "lawsuit", "legal", "unacceptable", "escalate", "demand", "threatening"
]

BUSINESS_IMPACT_KEYWORDS = [
    "refund", "chargeback", "cancel", "switch", "competitor", "never again",
    "will not recommend", "public review", "social media", "report"
]

HIGH_ISSUE_TYPES = {
    "Delivery Delay", "Missing / Lost Package", "Refund Not Processed",
    "Payment Issue", "Poor Customer Support"
}


def _score_single(row: dict) -> Tuple[int, List[str]]:
    """Calculate priority score for a single feedback row."""
    score = 0
    reasons = []

    # Sentiment
    sentiment = row.get("sentiment", "Neutral")
    s_score = SENTIMENT_WEIGHT.get(sentiment, 0)
    if s_score > 0:
        score += s_score
        reasons.append(f"Negative sentiment ({sentiment})")

    # Emotion
    emotion = row.get("emotion", "Neutral")
    e_score = EMOTION_WEIGHT.get(emotion, 0)
    if e_score > 0:
        score += e_score
        reasons.append(f"Strong negative emotion ({emotion})")

    # Is complaint
    if row.get("is_complaint", False):
        score += 2
        reasons.append("Complaint detected")

    # Issue type impact
    issue = row.get("detected_issue", "")
    if issue in HIGH_ISSUE_TYPES:
        score += 2
        reasons.append(f"High-impact issue type ({issue})")

    # Urgency keywords
    text = str(row.get("feedback", "")).lower()
    urgency_found = [kw for kw in URGENCY_KEYWORDS if kw in text]
    if urgency_found:
        score += len(urgency_found)
        reasons.append(f"Urgency signals: {', '.join(urgency_found[:2])}")

    # Business impact
    business_found = [kw for kw in BUSINESS_IMPACT_KEYWORDS if kw in text]
    if business_found:
        score += len(business_found)
        reasons.append(f"Business impact signals: {', '.join(business_found[:2])}")

    # Low rating
    try:
        rating = float(row.get("rating", 5))
        if rating <= 2:
            score += 2
            reasons.append(f"Very low rating ({rating})")
        elif rating <= 3:
            score += 1
    except (ValueError, TypeError):
        pass

    return score, reasons


def classify_priority(score: int) -> str:
    if score >= 7:
        return "HIGH"
    elif score >= 4:
        return "MEDIUM"
    else:
        return "LOW"


def score_priority(row: dict) -> Dict:
    """Public method to score priority for a single feedback dict."""
    score, reasons = _score_single(row)
    priority = classify_priority(score)
    return {
        "priority": priority,
        "priority_score": score,
        "priority_reasons": reasons
    }


def run(df: pd.DataFrame, feedback_col: str) -> pd.DataFrame:
    """Run priority scoring on the full dataframe."""
    records = df.to_dict("records")
    results = []
    for row in records:
        row["feedback"] = row.get(feedback_col, "")
        result = score_priority(row)
        results.append(result)

    priority_df = pd.DataFrame(results)
    df["priority"] = priority_df["priority"].values
    df["priority_score"] = priority_df["priority_score"].values
    df["priority_reasons"] = priority_df["priority_reasons"].values
    return df


def get_priority_distribution(df: pd.DataFrame) -> Dict:
    """Return counts per priority level."""
    if "priority" not in df.columns:
        return {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    counts = df["priority"].value_counts().to_dict()
    return {
        "HIGH": counts.get("HIGH", 0),
        "MEDIUM": counts.get("MEDIUM", 0),
        "LOW": counts.get("LOW", 0),
    }
