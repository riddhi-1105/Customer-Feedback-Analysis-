"""
Emotion Analysis Tool
=====================
Rule-based emotion detection (no paid API) with whole-word matching,
intensifiers, negation handling and sentiment-consistency correction.

Emotions: Happy, Satisfied, Neutral, Frustrated, Angry, Disappointed, Confused
"""

import re
from typing import Dict, Optional

import pandas as pd

# Trailing "*" = word prefix, otherwise whole word / phrase.
EMOTION_KEYWORDS = {
    "Angry": {
        "keywords": ["furious", "outraged", "infuriated", "enraged", "livid", "angry",
                     "terrible", "horrible", "disgusting", "unacceptable", "appalling",
                     "scam", "fraud", "cheat*", "worst", "hate", "ridiculous", "absurd",
                     "pathetic", "useless", "incompetent", "rude"],
        "weight": 3,
    },
    "Frustrated": {
        "keywords": ["frustrat*", "annoying", "irritating", "fed up", "keeps failing",
                     "keeps showing", "not working", "still no*", "still waiting",
                     "still hasn't", "again and again", "waste of time", "poor experience",
                     "no response", "no reply", "no resolution", "no fix", "never resolved",
                     "never got resolved", "second time", "needs serious improvement",
                     "hassle", "multiple times", "crashes", "crashed", "timed out",
                     "hasn't happened", "not been processed", "never showed up"],
        "weight": 2,
    },
    "Disappointed": {
        "keywords": ["disappoint*", "let down", "letdown", "expected better", "expected a bit more",
                     "not what i expected", "below expectations", "not satisfied",
                     "unsatisfied", "not happy", "poor quality", "not worth", "waste of money",
                     "regret", "unfortunately", "underwhelming", "forgettable", "flimsy",
                     "not great", "would not recommend", "wouldn't recommend",
                     "won't order again", "not planning to purchase", "not worth the price",
                     "overcharged", "took longer", "slower than"],
        "weight": 2,
    },
    "Confused": {
        "keywords": ["confused", "unclear", "not sure", "don't understand", "no idea",
                     "complicated", "hard to understand", "difficult to navigate", "confusing",
                     "misleading", "not clear", "vague", "how do i", "without any explanation",
                     "no explanation"],
        "weight": 2,
    },
    "Satisfied": {
        "keywords": ["satisfied", "happy with", "pleased", "decent", "okay", "alright",
                     "fine", "works", "good enough", "acceptable", "meets expectations",
                     "as expected", "delivered on time", "got my order", "polite", "courteous",
                     "smooth", "accurate", "reasonable", "eventually", "no complaints",
                     "gets the job done", "solid", "average"],
        "weight": 2,
    },
    "Happy": {
        "keywords": ["love*", "excellent", "amazing", "fantastic", "wonderful", "brilliant",
                     "superb", "outstanding", "perfect*", "great", "awesome", "best", "happy",
                     "delighted", "thrilled", "impressed", "incredible", "10/10", "five star",
                     "highly recommend", "will buy again", "top notch", "worth every",
                     "premium", "flawless", "fantastic"],
        "weight": 3,
    },
}

INTENSIFIERS = {"very", "extremely", "so", "really", "absolutely", "totally", "completely"}
NEGATION_RE = re.compile(r"\b(not|never|no|don't|doesn't|didn't|won't|isn't|wasn't|couldn't|can't|hasn't)\b")

_COMPILED = {}
for _emo, _cfg in EMOTION_KEYWORDS.items():
    pats = []
    for kw in _cfg["keywords"]:
        stem = kw.endswith("*")
        base = re.escape(kw.rstrip("*")).replace(r"\ ", r"\s+")
        pats.append(re.compile(r"(?<![\w'])" + base + (r"\w*" if stem else r"(?![\w'])")))
    _COMPILED[_emo] = pats


def _count_emotion_score(text_lower: str, emotion: str) -> float:
    words = re.findall(r"[\w']+", text_lower)
    score = 0.0
    for pat in _COMPILED[emotion]:
        m = pat.search(text_lower)
        if m:
            first = m.group(0).split()[0]
            idx = next((i for i, w in enumerate(words) if w.startswith(first)), -1)
            boost = 1.5 if idx > 0 and words[idx - 1] in INTENSIFIERS else 1.0
            score += boost
    return score


def detect_emotion(text: str) -> Dict:
    """Return the detected emotion and confidence for a single text."""
    if not isinstance(text, str) or len(text.strip()) < 3:
        return {"emotion": "Neutral", "emotion_confidence": 50.0}

    text_lower = text.lower().replace("\u2019", "'")
    scores = {e: _count_emotion_score(text_lower, e) * cfg["weight"]
              for e, cfg in EMOTION_KEYWORDS.items()}

    if NEGATION_RE.search(text_lower):
        scores["Happy"] *= 0.3
        scores["Satisfied"] *= 0.5
        scores["Frustrated"] *= 1.2
        scores["Disappointed"] *= 1.2

    max_emotion = max(scores, key=scores.get)
    max_score = scores[max_emotion]
    if max_score == 0:
        return {"emotion": "Neutral", "emotion_confidence": 55.0}

    total = sum(scores.values()) or 1
    confidence = round(min(95.0, 50 + (max_score / total) * 45), 1)
    return {"emotion": max_emotion, "emotion_confidence": confidence}


_NEG_EMOTIONS = {"Angry", "Frustrated", "Disappointed", "Confused"}
_POS_EMOTIONS = {"Happy", "Satisfied"}


def _reconcile(emotion: str, conf: float, sentiment: Optional[str]) -> (str, float):
    """Keep emotion consistent with the (rating-anchored) sentiment."""
    if sentiment == "Negative":
        if emotion in _POS_EMOTIONS or emotion == "Neutral":
            return "Disappointed", 60.0
    elif sentiment == "Positive":
        if emotion in _NEG_EMOTIONS:
            return "Satisfied", 60.0
        if emotion == "Neutral":
            return "Satisfied", 60.0
    elif sentiment == "Neutral":
        if emotion in _POS_EMOTIONS and conf < 70:
            return "Neutral", 60.0
    return emotion, conf


def run(df: pd.DataFrame, feedback_col: str) -> pd.DataFrame:
    """Run emotion detection on all feedback rows."""
    df = df.copy()
    sentiments = df["sentiment"].tolist() if "sentiment" in df.columns else [None] * len(df)
    emos, confs = [], []
    for text, sent in zip(df[feedback_col].astype(str), sentiments):
        r = detect_emotion(text)
        e, c = _reconcile(r["emotion"], r["emotion_confidence"], sent)
        emos.append(e)
        confs.append(c)
    df["emotion"] = emos
    df["emotion_confidence"] = confs
    return df
