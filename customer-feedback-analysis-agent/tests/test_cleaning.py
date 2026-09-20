"""
Tests for Data Cleaning Tool
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
from tools.data_cleaner import run, clean_text


class TestCleanText:
    def test_strips_whitespace(self):
        assert clean_text("  hello world  ") == "hello world"

    def test_normalizes_extra_spaces(self):
        assert clean_text("hello   world") == "hello world"

    def test_handles_none(self):
        assert clean_text(None) == ""

    def test_handles_non_string(self):
        assert clean_text(123) == ""

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_normal_text(self):
        result = clean_text("The product quality is excellent!")
        assert result == "The product quality is excellent!"


class TestDataCleaner:
    def setup_method(self):
        self.df = pd.DataFrame({
            "feedback": [
                "Great product, very happy!",
                "Terrible service, never again.",
                None,
                "   ",
                "Great product, very happy!",  # duplicate
                "ok",  # too short (< 3 words)
                "The delivery was very late and I am disappointed.",
                "Product quality is bad.",
            ]
        })

    def test_removes_nulls(self):
        clean_df, report = run(self.df, "feedback")
        assert report["missing_handled"] > 0

    def test_removes_duplicates(self):
        clean_df, report = run(self.df, "feedback")
        assert report["duplicates_removed"] >= 1

    def test_removes_short_feedback(self):
        clean_df, report = run(self.df, "feedback")
        assert report["short_removed"] >= 1

    def test_valid_records_added(self):
        clean_df, report = run(self.df, "feedback")
        assert report["valid_records"] > 0
        assert report["valid_records"] < report["records_received"]

    def test_feedback_id_assigned(self):
        clean_df, report = run(self.df, "feedback")
        assert "feedback_id" in clean_df.columns

    def test_index_reset(self):
        clean_df, report = run(self.df, "feedback")
        assert clean_df.index.tolist() == list(range(len(clean_df)))

    def test_quality_report_keys(self):
        clean_df, report = run(self.df, "feedback")
        required_keys = ["records_received", "duplicates_removed", "missing_handled", "valid_records"]
        for key in required_keys:
            assert key in report, f"Missing key: {key}"

    def test_records_received_correct(self):
        clean_df, report = run(self.df, "feedback")
        assert report["records_received"] == len(self.df)

    def test_empty_dataframe(self):
        empty_df = pd.DataFrame({"feedback": []})
        clean_df, report = run(empty_df, "feedback")
        assert report["valid_records"] == 0
