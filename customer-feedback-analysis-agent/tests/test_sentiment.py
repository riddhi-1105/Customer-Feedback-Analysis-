"""
Tests for Sentiment Analysis Tool
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
from tools.sentiment_tool import analyze_single, run


class TestSentimentSingle:
    def test_positive_feedback(self):
        result = analyze_single("I love this product! It is excellent and amazing.")
        assert result["sentiment"] == "Positive"
        assert result["sentiment_confidence"] > 50

    def test_negative_feedback(self):
        result = analyze_single("The product is terrible and I hate it. Worst purchase ever.")
        assert result["sentiment"] == "Negative"
        assert result["sentiment_confidence"] > 50

    def test_result_has_keys(self):
        result = analyze_single("The product is okay.")
        assert "sentiment" in result
        assert "sentiment_confidence" in result

    def test_sentiment_is_valid(self):
        texts = [
            "Excellent product!",
            "Average experience.",
            "Very bad service.",
        ]
        valid_sentiments = {"Positive", "Neutral", "Negative"}
        for text in texts:
            result = analyze_single(text)
            assert result["sentiment"] in valid_sentiments

    def test_confidence_range(self):
        result = analyze_single("Good product overall.")
        assert 0 <= result["sentiment_confidence"] <= 100

    def test_strong_positive(self):
        result = analyze_single("This is the best product I've ever bought! Absolutely love it!")
        assert result["sentiment"] == "Positive"

    def test_strong_negative(self):
        result = analyze_single("Terrible experience. Never buying again. Completely broken.")
        assert result["sentiment"] == "Negative"


class TestSentimentDataframe:
    def setup_method(self):
        self.df = pd.DataFrame({
            "feedback": [
                "Great product, very happy!",
                "Terrible service, very disappointed.",
                "The product is okay, nothing special.",
            ]
        })

    def test_adds_sentiment_column(self):
        result_df = run(self.df, "feedback")
        assert "sentiment" in result_df.columns

    def test_adds_confidence_column(self):
        result_df = run(self.df, "feedback")
        assert "sentiment_confidence" in result_df.columns

    def test_row_count_preserved(self):
        result_df = run(self.df, "feedback")
        assert len(result_df) == len(self.df)

    def test_all_sentiments_valid(self):
        result_df = run(self.df, "feedback")
        valid = {"Positive", "Neutral", "Negative"}
        assert set(result_df["sentiment"].unique()).issubset(valid)
