"""
Sentiment Analysis Tool
=======================
Hybrid sentiment engine that works out-of-the-box on any host (no runtime
downloads, no NLTK corpora):

1. RATING  - if the dataset has a 1-5 star rating, it is the most reliable
             signal a customer gives, so it is used directly
             (1-2 = Negative, 3 = Neutral, 4-5 = Positive).
2. TEXT    - otherwise the text is scored by an ensemble of
             a) a TF-IDF + Logistic-Regression model trained at start-up on the
                bundled labelled dataset (data/customer_feedback_dataset.csv)
             b) VADER (lexicon ships inside the `vaderSentiment` pip package)
                extended with a customer-feedback domain lexicon
3. Fallback - a tiny keyword scorer if neither library is importable.

Public API:  analyze_single(text) -> dict,  run(df, col, rating_col) -> df
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LABELS = ["Negative", "Neutral", "Positive"]
_ROOT = Path(__file__).resolve().parent.parent
_TRAIN_FILE = _ROOT / "data" / "customer_feedback_dataset.csv"

# ---------------------------------------------------------------------------
# VADER (bundled lexicon - nothing to download)
# ---------------------------------------------------------------------------
_VADER = None
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    _VADER = SentimentIntensityAnalyzer()
    # Domain vocabulary that generic VADER scores as neutral
    _VADER.lexicon.update({
        "flimsy": -2.2, "overcharged": -2.6, "unresolved": -2.0, "glitch": -1.6,
        "glitches": -1.6, "hiccup": -1.0, "mediocre": -1.4, "refund": -0.4,
        "delayed": -1.8, "delay": -1.4, "damaged": -2.4, "defective": -2.6,
        "wear": -0.8, "lagging": -1.6, "crashes": -2.2, "crashed": -2.2,
        "sturdy": 1.8, "premium": 1.8, "courteous": 1.8, "seamless": 2.0,
        "smooth": 1.6, "reliable": 1.8, "worth": 1.2, "polite": 0.8,
        "eventually": -0.6, "unhelpful": -2.2, "ignored": -2.2, "pathetic": -2.8,
        "scam": -3.0, "fraud": -3.0, "ripoff": -2.8, "misleading": -2.0,
    })
except Exception as exc:  # pragma: no cover
    logger.warning("vaderSentiment unavailable: %s", exc)

_VADER_AVAILABLE = _VADER is not None
_TEXTBLOB_AVAILABLE = False  # kept for backwards-compat; TextBlob is no longer used


def _vader_probs(text: str) -> np.ndarray:
    """Convert VADER compound score into [neg, neu, pos] probabilities."""
    c = _VADER.polarity_scores(text)["compound"]
    if c >= 0.05:
        p = np.array([0.05, 0.25 - 0.1 * c, 0.7 + 0.25 * c])
    elif c <= -0.05:
        p = np.array([0.7 + 0.25 * -c, 0.25 - 0.1 * -c, 0.05])
    else:
        p = np.array([0.2, 0.6, 0.2])
    p = np.clip(p, 0.01, None)
    return p / p.sum()


# ---------------------------------------------------------------------------
# Trained text model (lazy, cached in-process)
# ---------------------------------------------------------------------------
_MODEL = None
_MODEL_TRIED = False
MODEL_INFO: Dict = {"available": False}


def _load_model():
    """Train TF-IDF + LogisticRegression on the bundled labelled file (~0.5 s)."""
    global _MODEL, _MODEL_TRIED
    if _MODEL_TRIED:
        return _MODEL
    _MODEL_TRIED = True
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.pipeline import make_pipeline

        if not _TRAIN_FILE.exists():
            return None
        train = pd.read_csv(_TRAIN_FILE).dropna(subset=["feedback", "sentiment"])
        train = train.drop_duplicates(subset=["feedback"])
        train = train[train["sentiment"].isin(LABELS)]
        if len(train) < 60:
            return None

        pipe = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=True),
            LogisticRegression(C=5.0, max_iter=2000, class_weight="balanced"),
        )
        # Honest, held-out accuracy estimate (5-fold CV on unique texts)
        cv_pred = cross_val_predict(
            pipe, train["feedback"], train["sentiment"],
            cv=StratifiedKFold(5, shuffle=True, random_state=42),
        )
        acc = float((cv_pred == train["sentiment"]).mean())
        pipe.fit(train["feedback"], train["sentiment"])
        _MODEL = pipe
        MODEL_INFO.update({
            "available": True,
            "train_rows": int(len(train)),
            "cv_accuracy": round(acc * 100, 1),
            "classes": list(pipe.classes_),
        })
    except Exception as exc:  # pragma: no cover
        logger.warning("Sentiment model unavailable: %s", exc)
        _MODEL = None
    return _MODEL


def _model_probs(text: str) -> Optional[np.ndarray]:
    model = _load_model()
    if model is None:
        return None
    proba = model.predict_proba([text])[0]
    ordered = {c: p for c, p in zip(model.classes_, proba)}
    return np.array([ordered.get(lbl, 0.0) for lbl in LABELS])


# ---------------------------------------------------------------------------
# Keyword fallback
# ---------------------------------------------------------------------------
_POS = {"good", "great", "excellent", "love", "perfect", "amazing", "wonderful",
        "fantastic", "happy", "satisfied", "best", "recommend", "helpful", "fast",
        "quick", "easy", "smooth", "nice", "awesome", "sturdy", "premium"}
_NEG = {"bad", "worst", "terrible", "horrible", "awful", "hate", "slow", "broken",
        "useless", "disappointed", "angry", "frustrated", "never", "problem",
        "issue", "wrong", "delayed", "late", "missing", "damaged", "refund",
        "cancel", "failed", "error", "poor", "flimsy", "overcharged"}


def _keyword_probs(text: str) -> np.ndarray:
    words = set(re.findall(r"[a-z']+", text.lower()))
    pos, neg = len(words & _POS), len(words & _NEG)
    if pos > neg:
        return np.array([0.15, 0.2, 0.65])
    if neg > pos:
        return np.array([0.65, 0.2, 0.15])
    return np.array([0.2, 0.6, 0.2])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def text_probabilities(text: str) -> np.ndarray:
    """Ensemble probabilities [neg, neu, pos] from text only."""
    parts, weights = [], []
    m = _model_probs(text)
    if m is not None:
        parts.append(m)
        weights.append(0.7)
    if _VADER_AVAILABLE:
        parts.append(_vader_probs(text))
        weights.append(0.3 if m is not None else 1.0)
    if not parts:
        return _keyword_probs(text)
    w = np.array(weights) / sum(weights)
    return sum(wi * pi for wi, pi in zip(w, parts))


def analyze_single(text: str) -> Dict:
    """Analyze sentiment of a single feedback string (text only)."""
    text = str(text or "").strip()
    if not text:
        return {"sentiment": "Neutral", "sentiment_confidence": 50.0}
    probs = text_probabilities(text)
    idx = int(np.argmax(probs))
    return {
        "sentiment": LABELS[idx],
        "sentiment_confidence": round(float(probs[idx]) * 100, 1),
    }


def rating_to_sentiment(value) -> Optional[str]:
    """Map a 1-5 rating to a sentiment label (None if not usable)."""
    try:
        r = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(r) or r < 1 or r > 5:
        return None
    if r <= 2:
        return "Negative"
    if r < 4:
        return "Neutral"
    return "Positive"


def run(df: pd.DataFrame, feedback_col: str, rating_col: Optional[str] = None) -> pd.DataFrame:
    """
    Add `sentiment`, `sentiment_confidence`, `sentiment_source` columns.

    If the uploaded file already contains a `sentiment` column it is preserved
    as `sentiment_provided` so the app can report agreement with it.
    """
    df = df.copy()
    if "sentiment" in df.columns:
        df["sentiment_provided"] = df["sentiment"]

    labels, confs, sources = [], [], []
    use_rating = bool(rating_col and rating_col in df.columns)
    ratings = pd.to_numeric(df[rating_col], errors="coerce") if use_rating else None

    for i, text in enumerate(df[feedback_col].astype(str).tolist()):
        text_res = analyze_single(text)
        r_label = rating_to_sentiment(ratings.iloc[i]) if use_rating else None
        if r_label:
            # Rating is authoritative; confidence is higher when text agrees
            agree = text_res["sentiment"] == r_label
            labels.append(r_label)
            confs.append(95.0 if agree else 80.0)
            sources.append("rating")
        else:
            labels.append(text_res["sentiment"])
            confs.append(text_res["sentiment_confidence"])
            sources.append("text")

    df["sentiment"] = labels
    df["sentiment_confidence"] = confs
    df["sentiment_source"] = sources
    return df
