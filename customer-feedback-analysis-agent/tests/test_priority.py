"""
Tests for Priority Engine
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
from tools.priority_engine import score_priority, classify_priority, run, get_priority_distribution


class TestClassifyPriority:
    def test_high_priority(self):
        assert classify_priority(7) == "HIGH"
        assert classify_priority(10) == "HIGH"
        assert classify_priority(12) == "HIGH"

    def test_medium_priority(self):
        assert classify_priority(4) == "MEDIUM"
        assert classify_priority(5) == "MEDIUM"
        assert classify_priority(6) == "MEDIUM"

    def test_low_priority(self):
        assert classify_priority(0) == "LOW"
        assert classify_priority(2) == "LOW"
        assert classify_priority(3) == "LOW"


class TestScorePriority:
    def test_negative_sentiment_increases_score(self):
        result_neg = score_priority({"sentiment": "Negative", "emotion": "Neutral",
                                      "is_complaint": False, "detected_issue": "General Feedback", "feedback": ""})
        result_pos = score_priority({"sentiment": "Positive", "emotion": "Neutral",
                                      "is_complaint": False, "detected_issue": "General Feedback", "feedback": ""})
        assert result_neg["priority_score"] > result_pos["priority_score"]

    def test_angry_emotion_increases_score(self):
        result_angry = score_priority({"sentiment": "Negative", "emotion": "Angry",
                                        "is_complaint": True, "detected_issue": "Delivery Delay", "feedback": ""})
        result_happy = score_priority({"sentiment": "Positive", "emotion": "Happy",
                                        "is_complaint": False, "detected_issue": "General Feedback", "feedback": ""})
        assert result_angry["priority_score"] > result_happy["priority_score"]

    def test_complaint_increases_score(self):
        result_complaint = score_priority({"sentiment": "Negative", "emotion": "Neutral",
                                            "is_complaint": True, "detected_issue": "General Feedback", "feedback": ""})
        result_no_complaint = score_priority({"sentiment": "Negative", "emotion": "Neutral",
                                               "is_complaint": False, "detected_issue": "General Feedback", "feedback": ""})
        assert result_complaint["priority_score"] > result_no_complaint["priority_score"]

    def test_urgency_keywords_increase_score(self):
        result = score_priority({"sentiment": "Negative", "emotion": "Angry",
                                  "is_complaint": True, "detected_issue": "Delivery Delay",
                                  "feedback": "This is urgent and unacceptable!"})
        assert result["priority_score"] >= 7
        assert result["priority"] == "HIGH"

    def test_result_has_required_keys(self):
        result = score_priority({"sentiment": "Negative", "emotion": "Neutral",
                                  "is_complaint": True, "detected_issue": "General Feedback", "feedback": ""})
        assert "priority" in result
        assert "priority_score" in result
        assert "priority_reasons" in result

    def test_priority_is_valid(self):
        result = score_priority({"sentiment": "Positive", "emotion": "Happy",
                                  "is_complaint": False, "detected_issue": "General Feedback", "feedback": ""})
        assert result["priority"] in {"HIGH", "MEDIUM", "LOW"}

    def test_reasons_list(self):
        result = score_priority({"sentiment": "Negative", "emotion": "Angry",
                                  "is_complaint": True, "detected_issue": "Delivery Delay", "feedback": ""})
        assert isinstance(result["priority_reasons"], list)
        assert len(result["priority_reasons"]) > 0


class TestPriorityDataframe:
    def setup_method(self):
        self.df = pd.DataFrame({
            "feedback": [
                "Terrible experience! Urgent refund needed immediately.",
                "Great product, very happy!",
                "The delivery was late.",
            ],
            "sentiment": ["Negative", "Positive", "Negative"],
            "emotion": ["Angry", "Happy", "Frustrated"],
            "detected_issue": ["Refund Not Processed", "General Feedback", "Delivery Delay"],
            "is_complaint": [True, False, True],
        })

    def test_adds_priority_column(self):
        result_df = run(self.df, "feedback")
        assert "priority" in result_df.columns

    def test_adds_priority_score_column(self):
        result_df = run(self.df, "feedback")
        assert "priority_score" in result_df.columns

    def test_row_count_preserved(self):
        result_df = run(self.df, "feedback")
        assert len(result_df) == len(self.df)

    def test_priority_distribution(self):
        result_df = run(self.df, "feedback")
        dist = get_priority_distribution(result_df)
        assert "HIGH" in dist
        assert "MEDIUM" in dist
        assert "LOW" in dist
        assert sum(dist.values()) == len(result_df)
