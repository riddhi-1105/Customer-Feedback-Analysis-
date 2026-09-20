"""
Topic Detection Tool
====================
Classifies feedback into predefined topics using keyword matching + TF-IDF scoring.
Topics: Product Quality, Delivery, Pricing, Customer Support, Refund, Payment,
        Website/App, Product Features, Account, Other
"""

import re
import pandas as pd
from typing import Dict, List, Tuple

TOPIC_KEYWORDS = {
    "Delivery": [
        "delivery", "shipping", "shipped", "courier", "parcel", "package", "tracking",
        "arrived", "arrive", "late", "delayed", "delay", "days", "week", "transit",
        "dispatch", "out for delivery", "not received", "missing package", "lost package"
    ],
    "Customer Support": [
        "support", "customer service", "agent", "representative", "help", "helpline",
        "response", "respond", "reply", "chat", "call", "phone", "email", "complaint",
        "escalate", "team", "staff", "rude", "polite", "helpful", "useless agent",
        "no response", "no reply", "not picking up", "on hold"
    ],
    "Product Quality": [
        "quality", "defective", "broken", "damaged", "poor quality", "bad quality",
        "not working", "stopped working", "malfunctioning", "faulty", "cheap", "flimsy",
        "material", "build", "worn out", "peeling", "cracked", "torn"
    ],
    "Refund": [
        "refund", "return", "money back", "reimbursement", "refunded", "not refunded",
        "waiting for refund", "refund request", "return policy", "exchange"
    ],
    "Payment": [
        "payment", "charged", "charge", "billing", "invoice", "transaction", "deducted",
        "amount", "money deducted", "double charged", "extra charges", "payment failed",
        "payment issue", "payment gateway"
    ],
    "Pricing": [
        "price", "expensive", "costly", "overpriced", "cheap", "value for money",
        "affordable", "discount", "offer", "coupon", "promo", "deal", "cost", "worth"
    ],
    "Website/App": [
        "app", "website", "site", "page", "loading", "crash", "error", "bug", "login",
        "interface", "ui", "ux", "navigation", "search", "filter", "slow website",
        "app not working", "404", "blank page", "not loading"
    ],
    "Product Features": [
        "feature", "functionality", "option", "setting", "button", "missing feature",
        "wish it had", "should have", "lacks", "lacking", "no option", "can't",
        "specification", "spec", "color", "size", "variant"
    ],
    "Account": [
        "account", "login", "sign in", "sign up", "password", "reset", "otp", "verify",
        "profile", "username", "registered", "registration", "log out", "session"
    ],
}


def detect_topic(text: str) -> Tuple[str, float]:
    """Return best matching topic and a simple confidence score."""
    text_lower = text.lower()
    topic_scores: Dict[str, int] = {}

    for topic, keywords in TOPIC_KEYWORDS.items():
        score = 0
        for kw in keywords:
            if kw in text_lower:
                # Multi-word keywords score higher
                score += len(kw.split())
        if score > 0:
            topic_scores[topic] = score

    if not topic_scores:
        return "Other", 50.0

    best_topic = max(topic_scores, key=topic_scores.get)
    best_score = topic_scores[best_topic]
    total = sum(topic_scores.values()) or 1
    confidence = round(min(95.0, 50 + (best_score / total) * 80), 1)
    return best_topic, confidence


def run(df: pd.DataFrame, feedback_col: str) -> pd.DataFrame:
    """Run topic detection on all feedback rows."""
    results = df[feedback_col].apply(
        lambda t: pd.Series({"topic": detect_topic(str(t))[0],
                              "topic_confidence": detect_topic(str(t))[1]})
    )
    df = df.copy()
    for col in results.columns:
        df[col] = results[col].values
    return df


def get_topic_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Return topic counts for chart display."""
    if "topic" not in df.columns:
        return pd.DataFrame()
    return df["topic"].value_counts().reset_index().rename(
        columns={"index": "topic", "topic": "count"}
    )
