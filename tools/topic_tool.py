"""
Topic Detection Tool
====================
Classifies feedback into business topics using whole-word keyword scoring.

* Whole-word / word-prefix matching (the old substring matching made "app"
  match inside "happy" and "ui" inside "quite").
* If the uploaded file already contains a curated category column
  (e.g. ``feedback_category``) the agent uses that instead and only reports
  how well the automatic detector agrees with it.
"""

import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

# A trailing "*" means "word prefix" (deliver* -> delivery, delivered ...).
# Without it the keyword must match a whole word / phrase.
TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "Delivery": [
        "deliver*", "shipping", "shipped", "shipment", "courier", "parcel*", "package*",
        "tracking", "arrived", "arrive", "arrival", "late", "delayed", "delay",
        "transit", "dispatch*", "pickup", "pick up", "not received|3", "never received",
        "haven't received|3", "hasn't arrived|2", "left it at", "order confirmation|0",
    ],
    "Customer Support": [
        "support|2", "customer service|3", "customer care|3", "agent*|2", "representative*",
        "helpline", "respond*", "response", "reply", "replied", "escalat*", "staff",
        "rude", "polite", "courteous", "on hold", "transferred|2", "no resolution|2",
        "resolve*", "explain my issue|3", "contacting", "follow-ups",
    ],
    "Product Quality": [
        "quality|2", "defect*|2", "broken", "broke", "damaged", "not working", "stopped working",
        "stopped functioning|2", "malfunction*", "faulty", "cheap", "flimsy|2", "material*",
        "build|2", "worn", "wear", "peeling", "cracked", "torn", "sturdy|2", "well made|2",
        "durable|2", "finish", "scratch*", "scuff*|2", "crushed", "premium|2", "solid|2",
        "holding up|3", "as advertised|2", "matches the description|3", "feels|1",
    ],
    "Refund": [
        "refund*|3", "return|2", "returns|2", "returned|2", "money back|3", "reimburse*|3",
        "exchange", "return label|3", "sent back|2",
    ],
    "Payment": [
        "payment|2", "charged|2", "charge", "charges", "billing|2", "billed|2", "invoice|2",
        "transaction", "deducted|3", "amount", "double charged|4", "charged twice|4",
        "gateway|2", "card", "receipt", "overcharged|3", "upi", "emi", "bill|2",
        "checkout|1", "confirmation email|2", "confirmation|1",
    ],
    "Pricing": [
        "price", "prices", "expensive", "costly", "overpriced", "value for money",
        "affordable", "discount*", "coupon*", "promo*", "deal", "cost", "worth",
        "rupee*", "budget", "pricing",
    ],
    "Website/App": [
        "app|2", "apps|2", "website|2", "site", "page", "loading", "loads", "crash*|2", "error",
        "bug", "glitch*", "interface", "navigation", "search", "checkout", "laggy|2",
        "timed out|2", "notifications", "not loading", "slow website",
    ],
    "Product Features": [
        "feature*", "functionality", "option", "setting*", "button", "missing feature",
        "wish it had", "should have", "lacks", "lacking", "no option", "specification*",
        "colour", "color", "size", "variant",
    ],
    "Account": [
        "account|2", "login|2", "log in|2", "sign in|2", "sign up", "password|2", "reset", "otp|2",
        "verify", "verification", "profile", "username", "registered", "registration",
        "log out", "locked|2",
    ],
}

# Compile once
_COMPILED: Dict[str, List[Tuple[re.Pattern, int]]] = {}
for _topic, _kws in TOPIC_KEYWORDS.items():
    _pats = []
    for _kw in _kws:
        _kw, _, _w = _kw.partition("|")
        _weight = int(_w) if _w else max(1, len(_kw.rstrip("*").split()))
        _stem = _kw.endswith("*")
        _base = re.escape(_kw.rstrip("*")).replace(r"\ ", r"\s+")
        _pats.append((re.compile(r"\b" + _base + (r"\w*" if _stem else r"\b")), _weight))
    _COMPILED[_topic] = _pats

_CATEGORY_CANDIDATES = [
    "feedback_category", "category", "topic", "issue_category", "complaint_category",
    "department", "type", "feedback_type", "tag", "theme",
]


def detect_category_column(df: pd.DataFrame, feedback_col: Optional[str] = None) -> Optional[str]:
    """Find an existing low-cardinality category column in the uploaded data."""
    lower = {str(c).lower(): c for c in df.columns}
    for cand in _CATEGORY_CANDIDATES:
        col = lower.get(cand)
        if col is None or col == feedback_col:
            continue
        s = df[col].dropna().astype(str)
        if len(s) and 2 <= s.nunique() <= 30 and s.str.len().mean() < 40:
            return col
    return None


def detect_topic(text: str) -> Tuple[str, float]:
    """Return best matching topic and a confidence score (0-100)."""
    text_lower = str(text).lower().replace("\u2019", "'")
    topic_scores: Dict[str, int] = {}

    for topic, pats in _COMPILED.items():
        score = 0
        for pat, weight in pats:
            if pat.search(text_lower):
                score += weight  # strong / multi-word phrases are stronger evidence
        if score > 0:
            topic_scores[topic] = score

    if not topic_scores:
        return "Other", 50.0

    best_topic = max(topic_scores, key=topic_scores.get)
    best_score = topic_scores[best_topic]
    total = sum(topic_scores.values()) or 1
    confidence = round(min(95.0, 50 + (best_score / total) * 45), 1)
    return best_topic, confidence


def run(df: pd.DataFrame, feedback_col: str, category_col: Optional[str] = None) -> pd.DataFrame:
    """Add `topic` and `topic_confidence` (plus `topic_source`)."""
    df = df.copy()
    texts = df[feedback_col].astype(str).tolist()
    detected = [detect_topic(t) for t in texts]
    df["topic_detected"] = [d[0] for d in detected]

    if category_col and category_col in df.columns:
        provided = df[category_col].astype(str).str.strip()
        valid = provided.notna() & (provided != "") & (provided.str.lower() != "nan")
        df["topic"] = provided.where(valid, df["topic_detected"])
        df["topic_confidence"] = [98.0 if v else d[1] for v, d in zip(valid, detected)]
        df["topic_source"] = ["dataset" if v else "detected" for v in valid]
    else:
        df["topic"] = df["topic_detected"]
        df["topic_confidence"] = [d[1] for d in detected]
        df["topic_source"] = "detected"
    return df


def get_topic_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Return topic counts for chart display."""
    if "topic" not in df.columns:
        return pd.DataFrame()
    out = df["topic"].value_counts().reset_index()
    out.columns = ["topic", "count"]
    return out
