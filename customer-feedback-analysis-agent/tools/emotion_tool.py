"""
Emotion Analysis Tool
=====================
Detects customer emotion from feedback text using keyword/rule-based approach
with contextual intensifiers. No paid API needed.

Emotions: Happy, Satisfied, Neutral, Frustrated, Angry, Disappointed, Confused
"""

import re
import pandas as pd
from typing import Dict


# Keyword dictionaries for each emotion
EMOTION_KEYWORDS = {
    "Angry": {
        "keywords": ["furious", "outraged", "infuriated", "enraged", "livid", "angry",
                     "terrible", "horrible", "disgusting", "unacceptable", "appalling",
                     "scam", "fraud", "cheat", "worst", "hate", "ridiculous", "absurd",
                     "pathetic", "useless", "incompetent"],
        "weight": 3
    },
    "Frustrated": {
        "keywords": ["frustrated", "annoying", "irritating", "fed up", "disappointed with",
                     "keeps failing", "not working", "still not resolved", "again and again",
                     "waste of time", "pathetic service", "poor experience", "broken",
                     "delayed", "late", "waiting", "hours", "days", "waiting forever"],
        "weight": 2
    },
    "Disappointed": {
        "keywords": ["disappointed", "let down", "expected better", "not what i expected",
                     "below expectations", "not satisfied", "unsatisfied", "not happy",
                     "poor quality", "not worth", "waste of money", "regret", "unfortunately"],
        "weight": 2
    },
    "Confused": {
        "keywords": ["confused", "unclear", "not sure", "don't understand", "no idea",
                     "complicated", "hard to understand", "difficult to navigate", "confusing",
                     "misleading", "not clear", "vague", "where", "how do i", "why"],
        "weight": 2
    },
    "Satisfied": {
        "keywords": ["satisfied", "happy with", "pleased", "decent", "okay", "alright",
                     "fine", "works", "good enough", "acceptable", "meets expectations",
                     "as expected", "delivered on time", "got my order"],
        "weight": 2
    },
    "Happy": {
        "keywords": ["love", "excellent", "amazing", "fantastic", "wonderful", "brilliant",
                     "superb", "outstanding", "perfect", "great", "awesome", "best", "happy",
                     "delighted", "thrilled", "impressed", "incredible", "10/10", "five star",
                     "highly recommend", "will buy again", "top notch"],
        "weight": 3
    },
}

INTENSIFIERS = {"very", "extremely", "so", "really", "absolutely", "totally", "completely"}
NEGATIONS = {"not", "never", "no", "don't", "doesn't", "didn't", "won't", "isn't", "wasn't"}


def _count_emotion_score(text: str, keywords: list) -> float:
    """Count keyword matches with intensifier boost."""
    text_lower = text.lower()
    words = re.findall(r"\b\w+\b", text_lower)
    score = 0.0
    for kw in keywords:
        if kw in text_lower:
            # Boost if preceded by intensifier
            kw_words = kw.split()
            idx = next((i for i, w in enumerate(words) if w == kw_words[0]), -1)
            boost = 1.5 if idx > 0 and words[idx - 1] in INTENSIFIERS else 1.0
            score += boost
    return score


def detect_emotion(text: str) -> Dict:
    """Return the detected emotion and confidence for a single text."""
    if not isinstance(text, str) or len(text.strip()) < 3:
        return {"emotion": "Neutral", "emotion_confidence": 50.0}

    scores = {}
    for emotion, config in EMOTION_KEYWORDS.items():
        raw = _count_emotion_score(text, config["keywords"])
        scores[emotion] = raw * config["weight"]

    # Check for negation context
    text_lower = text.lower()
    has_negation = any(neg in text_lower for neg in NEGATIONS)
    if has_negation:
        # Flip Happy/Satisfied scores down, boost negative emotions
        scores["Happy"] *= 0.3
        scores["Satisfied"] *= 0.5
        scores["Frustrated"] *= 1.2
        scores["Disappointed"] *= 1.2

    max_emotion = max(scores, key=scores.get)
    max_score = scores[max_emotion]

    if max_score == 0:
        # Fall back based on simple polarity
        return {"emotion": "Neutral", "emotion_confidence": 55.0}

    total = sum(scores.values()) or 1
    confidence = round(min(95.0, 50 + (max_score / total) * 100), 1)
    return {"emotion": max_emotion, "emotion_confidence": confidence}


def run(df: pd.DataFrame, feedback_col: str) -> pd.DataFrame:
    """Run emotion detection on all feedback rows."""
    results = df[feedback_col].apply(lambda t: pd.Series(detect_emotion(str(t))))
    df = df.copy()
    for col in results.columns:
        df[col] = results[col].values
    return df
