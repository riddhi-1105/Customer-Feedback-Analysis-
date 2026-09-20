"""
Recurring Issue Detection Tool
================================
Groups similar complaints using TF-IDF + cosine similarity clustering.
Falls back to keyword-based grouping if sklearn not available.

Returns a list of recurring issue clusters with counts and priority.
"""

import pandas as pd
import numpy as np
from typing import List, Dict

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import DBSCAN
    from sklearn.metrics.pairwise import cosine_similarity
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


RECURRING_THRESHOLD = 3  # Minimum occurrences to be "recurring"


def _cluster_with_tfidf(texts: List[str], eps: float = 0.35) -> List[int]:
    """Cluster texts using TF-IDF vectors + DBSCAN."""
    if len(texts) < 2:
        return list(range(len(texts)))

    vectorizer = TfidfVectorizer(max_features=500, stop_words="english", ngram_range=(1, 2))
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
    except Exception:
        return [-1] * len(texts)

    # Cosine distance
    cosine_sim = cosine_similarity(tfidf_matrix)
    distance_matrix = 1 - cosine_sim
    distance_matrix = np.clip(distance_matrix, 0, None)

    db = DBSCAN(eps=eps, min_samples=2, metric="precomputed")
    labels = db.fit_predict(distance_matrix)
    return labels.tolist()


def _keyword_cluster(texts: List[str]) -> List[int]:
    """Simple keyword-based grouping fallback."""
    ISSUE_GROUPS = {
        0: ["delivery", "shipping", "arrived late", "delayed", "parcel", "package", "transit"],
        1: ["refund", "return", "money back", "reimbursement"],
        2: ["payment", "charged", "billing", "transaction", "deducted"],
        3: ["support", "customer service", "agent", "response", "reply"],
        4: ["quality", "defective", "broken", "damaged", "faulty"],
        5: ["app", "website", "loading", "crash", "error", "bug"],
        6: ["account", "login", "password", "sign in"],
    }
    labels = []
    for text in texts:
        text_lower = text.lower()
        assigned = -1
        for group_id, keywords in ISSUE_GROUPS.items():
            if any(kw in text_lower for kw in keywords):
                assigned = group_id
                break
        labels.append(assigned)
    return labels


def _label_cluster(texts: List[str], cluster_id: int) -> str:
    """Generate a human-readable label for a cluster by majority keyword."""
    LABEL_MAP = {
        "delivery": "Delivery Delay",
        "shipping": "Delivery Delay",
        "refund": "Refund Not Processed",
        "payment": "Payment Issue",
        "support": "Poor Customer Support",
        "quality": "Poor Product Quality",
        "damaged": "Damaged Product",
        "app": "App / Website Issue",
        "website": "App / Website Issue",
        "account": "Account Issue",
        "cancel": "Order Cancellation",
        "wrong": "Wrong Product Received",
        "missing": "Missing / Lost Package",
    }
    keyword_counts: Dict[str, int] = {}
    for text in texts:
        text_lower = text.lower()
        for kw, label in LABEL_MAP.items():
            if kw in text_lower:
                keyword_counts[label] = keyword_counts.get(label, 0) + 1

    if keyword_counts:
        return max(keyword_counts, key=keyword_counts.get)
    return f"Issue Cluster {cluster_id + 1}"


def run(df: pd.DataFrame, feedback_col: str) -> List[Dict]:
    """
    Detect recurring issues using clustering.

    Returns list of dicts:
    {issue_label, occurrences, percentage, avg_sentiment, priority, sample_feedbacks}
    """
    texts = df[feedback_col].astype(str).tolist()

    if _SKLEARN_AVAILABLE:
        labels = _cluster_with_tfidf(texts)
    else:
        labels = _keyword_cluster(texts)

    # Build clusters
    clusters: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels):
        if label == -1:
            continue
        clusters.setdefault(label, []).append(idx)

    # Filter to recurring (>= threshold)
    recurring = []
    total = len(df)

    for cluster_id, indices in clusters.items():
        if len(indices) < RECURRING_THRESHOLD:
            continue

        cluster_texts = [texts[i] for i in indices]
        label = _label_cluster(cluster_texts, cluster_id)

        # Average sentiment
        avg_sentiment = "Neutral"
        if "sentiment" in df.columns:
            sentiments = df.iloc[indices]["sentiment"].value_counts()
            avg_sentiment = sentiments.index[0] if len(sentiments) > 0 else "Neutral"

        # Priority based on frequency + sentiment
        count = len(indices)
        percentage = round(count / total * 100, 1)
        if count >= 20 or avg_sentiment == "Negative":
            priority = "HIGH"
        elif count >= 10:
            priority = "MEDIUM"
        else:
            priority = "LOW"

        recurring.append({
            "issue_label": label,
            "occurrences": count,
            "percentage": percentage,
            "avg_sentiment": avg_sentiment,
            "priority": priority,
            "sample_feedbacks": cluster_texts[:3],
            "cluster_id": cluster_id,
        })

    # Sort by occurrences
    recurring.sort(key=lambda x: x["occurrences"], reverse=True)
    return recurring
