"""
Recurring Issue Detection Tool
==============================
Finds problems that keep coming back.

Only *complaints* are analysed (negative sentiment / detected complaint) -
clustering happy reviews and calling them "issues" was a bug in the first
version.

Method
------
1. Select complaint rows.
2. Assign each to an issue label (issue_detector's label if available,
   otherwise it is detected on the fly).
3. TF-IDF + DBSCAN (cosine) inside every issue group to find the distinct
   *wordings/themes* customers use for the same problem.
4. Rank issues by volume, severity and (if present) rating, and report which
   product / channel is affected most.

Falls back to plain grouping when scikit-learn is not importable.
"""

from collections import Counter
from typing import Dict, List

import numpy as np
import pandas as pd

from tools import issue_detector

try:
    from sklearn.cluster import DBSCAN
    from sklearn.feature_extraction.text import TfidfVectorizer
    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SKLEARN_AVAILABLE = False

RECURRING_THRESHOLD = 3          # minimum occurrences to be "recurring"
HIGH_IMPACT = {"Delivery Delay", "Missing / Lost Package", "Refund Not Processed",
               "Payment Issue", "Poor Customer Support", "Pricing / Overcharge"}
_SEGMENT_COLS = ["product", "channel", "topic"]


def _top_terms(texts: List[str], n: int = 4) -> List[str]:
    """Most characteristic terms of a group of texts (TF-IDF mass)."""
    if not _SKLEARN_AVAILABLE or not texts:
        return []
    try:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=300)
        m = vec.fit_transform(texts)
        scores = np.asarray(m.sum(axis=0)).ravel()
        terms = np.array(vec.get_feature_names_out())
        out = []
        for i in scores.argsort()[::-1]:
            t = terms[i]
            # avoid near-duplicate unigram/bigram pairs
            if not any(t in o or o in t for o in out):
                out.append(t)
            if len(out) >= n:
                break
        return out
    except Exception:
        return []


def _sub_themes(texts: List[str], eps: float = 0.55) -> List[Dict]:
    """Distinct wordings inside one issue group (DBSCAN on TF-IDF cosine)."""
    counts = Counter(texts)
    uniq = list(counts)
    if len(uniq) < 2 or not _SKLEARN_AVAILABLE:
        return [{"example": t, "count": c} for t, c in counts.most_common(3)]
    try:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        m = vec.fit_transform(uniq)
        dist = np.clip(1 - (m @ m.T).toarray(), 0, None)
        labels = DBSCAN(eps=eps, min_samples=1, metric="precomputed").fit_predict(dist)
    except Exception:
        return [{"example": t, "count": c} for t, c in counts.most_common(3)]
    groups: Dict[int, List[str]] = {}
    for lab, t in zip(labels, uniq):
        groups.setdefault(int(lab), []).append(t)
    themes = []
    for members in groups.values():
        total = sum(counts[t] for t in members)
        best = max(members, key=lambda t: counts[t])
        themes.append({"example": best, "count": total})
    return sorted(themes, key=lambda x: x["count"], reverse=True)[:3]


def _complaint_mask(df: pd.DataFrame) -> pd.Series:
    if "is_complaint" in df.columns:
        return df["is_complaint"].astype(bool)
    if "sentiment" in df.columns:
        return df["sentiment"] == "Negative"
    return pd.Series(True, index=df.index)


def run(df: pd.DataFrame, feedback_col: str) -> List[Dict]:
    """
    Detect recurring issues.

    Returns list of dicts:
    {issue_label, occurrences, percentage, avg_sentiment, priority,
     sample_feedbacks, keywords, themes, avg_rating, top_product, top_channel}
    """
    if df is None or len(df) == 0 or feedback_col not in df.columns:
        return []

    total = len(df)
    comp = df[_complaint_mask(df)].copy()
    if len(comp) < RECURRING_THRESHOLD:
        return []

    if "detected_issue" in comp.columns:
        labels = comp["detected_issue"].tolist()
    else:
        labels = [issue_detector.detect_issue(t, "Negative")[0] for t in comp[feedback_col].astype(str)]
    comp["_issue"] = labels
    comp = comp[comp["_issue"] != "General Feedback"]

    rating_col = next((c for c in ("rating", "score", "stars") if c in comp.columns), None)
    recurring = []

    for cid, (label, grp) in enumerate(comp.groupby("_issue", sort=False)):
        n = len(grp)
        if n < RECURRING_THRESHOLD:
            continue
        texts = grp[feedback_col].astype(str).tolist()

        avg_sentiment = "Negative"
        if "sentiment" in grp.columns and len(grp):
            avg_sentiment = grp["sentiment"].value_counts().index[0]

        share = n / total
        avg_rating = None
        if rating_col:
            r = pd.to_numeric(grp[rating_col], errors="coerce").mean()
            avg_rating = None if pd.isna(r) else round(float(r), 2)

        if n >= 10 or share >= 0.08 or (label in HIGH_IMPACT and n >= 6):
            priority = "HIGH"
        elif n >= 5 or share >= 0.04:
            priority = "MEDIUM"
        else:
            priority = "LOW"

        item = {
            "issue_label": label,
            "occurrences": int(n),
            "percentage": round(share * 100, 1),
            "avg_sentiment": avg_sentiment,
            "priority": priority,
            "sample_feedbacks": [t for t, _ in Counter(texts).most_common(3)],
            "keywords": _top_terms(texts),
            "themes": _sub_themes(texts),
            "avg_rating": avg_rating,
            "cluster_id": cid,
        }
        for seg in ("product", "channel"):
            if seg in grp.columns and grp[seg].notna().any():
                vc = grp[seg].value_counts()
                item[f"top_{seg}"] = str(vc.index[0])
                item[f"top_{seg}_count"] = int(vc.iloc[0])
        recurring.append(item)

    recurring.sort(key=lambda x: (x["occurrences"], x["priority"] == "HIGH"), reverse=True)
    return recurring
