"""
Sentiment Analysis Tool
=======================
Uses VADER (valence-aware dictionary) + TextBlob for robust sentiment classification.
Returns: Positive / Neutral / Negative + confidence score.

Designed to be non-blocking: if VADER lexicon is not yet downloaded,
falls back to TextBlob, then keyword-based analysis.
"""

import pandas as pd
from typing import Dict, Tuple

# ---------------------------------------------------------------------------
# Module-level availability flags
# ---------------------------------------------------------------------------
_VADER_AVAILABLE = False
_VADER_ANALYZER = None
_TEXTBLOB_AVAILABLE = False
_TextBlob = None

# Try to initialize VADER without blocking
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer as _VADER_CLS
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
        _VADER_ANALYZER = _VADER_CLS()
        _VADER_AVAILABLE = True
    except LookupError:
        # vader_lexicon not yet downloaded — will use TextBlob fallback
        pass
except Exception:
    pass

# Try the standalone vaderSentiment package (ships its own lexicon, so it needs
# no runtime download — this is what runs on Vercel's read-only filesystem).
if not _VADER_AVAILABLE:
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as _VS_CLS
        _VADER_ANALYZER = _VS_CLS()
        _VADER_AVAILABLE = True
    except Exception:
        pass

# Try TextBlob
try:
    from textblob import TextBlob as _TextBlobCls
    _TextBlob = _TextBlobCls
    _TEXTBLOB_AVAILABLE = True
except Exception:
    pass


def _ensure_vader():
    """Try to download VADER if needed (called lazily on first use)."""
    global _VADER_AVAILABLE, _VADER_ANALYZER
    if _VADER_AVAILABLE:
        return True
    # 1) Standalone vaderSentiment — no download, works on read-only filesystems.
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as _VS_CLS
        _VADER_ANALYZER = _VS_CLS()
        _VADER_AVAILABLE = True
        return True
    except Exception:
        pass
    # 2) NLTK VADER (local dev) — may need to download the lexicon.
    try:
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer as _VADER_CLS
        nltk.download("vader_lexicon", quiet=True)
        _VADER_ANALYZER = _VADER_CLS()
        _VADER_AVAILABLE = True
        return True
    except Exception:
        return False


def _vader_sentiment(text: str) -> Tuple[str, float]:
    scores = _VADER_ANALYZER.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        label = "Positive"
        confidence = min(1.0, (compound + 1) / 2)
    elif compound <= -0.05:
        label = "Negative"
        confidence = min(1.0, (-compound + 1) / 2)
    else:
        label = "Neutral"
        confidence = 1.0 - abs(compound)
    return label, round(confidence * 100, 1)


def _textblob_sentiment(text: str) -> Tuple[str, float]:
    polarity = _TextBlob(text).sentiment.polarity
    if polarity > 0.05:
        label = "Positive"
        confidence = min(100.0, 50 + polarity * 50)
    elif polarity < -0.05:
        label = "Negative"
        confidence = min(100.0, 50 + abs(polarity) * 50)
    else:
        label = "Neutral"
        confidence = 60.0
    return label, round(confidence, 1)


def _keyword_sentiment(text: str) -> Tuple[str, float]:
    """Last-resort keyword-based fallback."""
    text_lower = text.lower()
    positive_words = {"good", "great", "excellent", "love", "perfect", "amazing",
                      "wonderful", "fantastic", "happy", "satisfied", "best", "recommend",
                      "helpful", "fast", "quick", "easy", "smooth", "nice", "awesome"}
    negative_words = {"bad", "worst", "terrible", "horrible", "awful", "hate",
                      "slow", "broken", "useless", "disappointed", "angry", "frustrated",
                      "never", "problem", "issue", "wrong", "delayed", "late", "missing",
                      "damaged", "refund", "cancel", "failed", "error", "poor"}
    words = set(text_lower.split())
    pos = len(words & positive_words)
    neg = len(words & negative_words)
    if pos > neg:
        return "Positive", 65.0
    elif neg > pos:
        return "Negative", 65.0
    else:
        return "Neutral", 55.0


def analyze_single(text: str) -> Dict:
    """Analyze sentiment of a single feedback string."""
    if _VADER_AVAILABLE and _VADER_ANALYZER:
        label, confidence = _vader_sentiment(text)
    elif _TEXTBLOB_AVAILABLE and _TextBlob:
        label, confidence = _textblob_sentiment(text)
    else:
        label, confidence = _keyword_sentiment(text)
    return {"sentiment": label, "sentiment_confidence": confidence}


def run(df: pd.DataFrame, feedback_col: str) -> pd.DataFrame:
    """Run sentiment analysis on all feedback rows."""
    results = df[feedback_col].apply(lambda t: pd.Series(analyze_single(str(t))))
    df = df.copy()
    for col in results.columns:
        df[col] = results[col].values
    return df
