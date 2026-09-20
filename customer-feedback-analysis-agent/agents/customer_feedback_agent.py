"""
Customer Feedback Agent
========================
Main orchestration agent that coordinates all specialized tools.
Maintains a visible activity log of all actions performed.

Architecture:
CustomerFeedbackAgent
│
├── Data Cleaning Tool
├── Sentiment Analysis Tool
├── Emotion Analysis Tool
├── Topic Detection Tool
├── Issue Detector Tool
├── Recurring Issue Detection Tool
├── Priority Scoring Tool
├── Trend Analysis Tool
├── Insight Generation Tool
├── Recommendation Tool
└── Report Generation Tool
"""

import time
import traceback
import pandas as pd
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

# Import all tools
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import (
    data_cleaner,
    sentiment_tool,
    emotion_tool,
    topic_tool,
    issue_detector,
    recurring_issue_detector,
    priority_engine,
    trend_analyzer,
    insight_generator,
    recommendation_engine,
    report_generator,
)


@dataclass
class AgentActivity:
    """Single activity log entry."""
    step: int
    tool: str
    description: str
    status: str  # pending | running | done | error
    timestamp: str = ""
    duration_ms: int = 0
    details: str = ""

    def to_dict(self) -> Dict:
        return {
            "step": self.step,
            "tool": self.tool,
            "description": self.description,
            "status": self.status,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
            "details": self.details,
        }


class CustomerFeedbackAgent:
    """
    Main orchestration agent for customer feedback analysis.
    
    The agent autonomously selects and orchestrates tools based on input data,
    maintaining a full activity log visible to the user.
    """

    def __init__(self, progress_callback: Optional[Callable] = None):
        self.activity_log: List[AgentActivity] = []
        self.results: Dict = {}
        self._step_counter = 0
        self._progress_callback = progress_callback  # called with (step, total, message)

    def _log(self, tool: str, description: str, status: str = "done",
             details: str = "", duration_ms: int = 0) -> AgentActivity:
        self._step_counter += 1
        activity = AgentActivity(
            step=self._step_counter,
            tool=tool,
            description=description,
            status=status,
            timestamp=datetime.now().strftime("%H:%M:%S"),
            duration_ms=duration_ms,
            details=details,
        )
        self.activity_log.append(activity)
        return activity

    def _notify(self, message: str, step: int = 0, total: int = 12):
        if self._progress_callback:
            try:
                self._progress_callback(step, total, message)
            except Exception:
                pass

    def analyze_dataset(
        self,
        df: pd.DataFrame,
        feedback_col: str,
        date_col: Optional[str] = None,
        rating_col: Optional[str] = None,
    ) -> Dict:
        """
        Full agentic analysis pipeline for a dataset.
        
        Returns comprehensive results dict.
        """
        self.activity_log = []
        self.results = {}
        self._step_counter = 0
        total_steps = 12

        # ── Step 1: Observe Input ─────────────────────────────────────────────
        self._notify("Observing input data...", 1, total_steps)
        self._log("Input Observer", 
                  f"Dataset received — {len(df):,} records, {len(df.columns)} columns",
                  details=f"Columns: {', '.join(df.columns.tolist())}")

        # ── Step 2: Understand Data ───────────────────────────────────────────
        self._notify("Understanding data structure...", 2, total_steps)
        has_date = date_col and date_col in df.columns
        has_rating = rating_col and rating_col in df.columns
        self._log("Data Inspector",
                  f"Data structure understood — feedback column: '{feedback_col}', "
                  f"date: {'✓' if has_date else '✗'}, rating: {'✓' if has_rating else '✗'}")

        # ── Step 3: Data Cleaning ─────────────────────────────────────────────
        self._notify("Cleaning data...", 3, total_steps)
        t0 = time.time()
        try:
            df_clean, quality_report = data_cleaner.run(df, feedback_col)
            duration = int((time.time() - t0) * 1000)
            self._log("Data Cleaner",
                      f"Data cleaned — {quality_report['valid_records']:,} valid records from "
                      f"{quality_report['records_received']:,} received, "
                      f"{quality_report['duplicates_removed']} duplicates removed",
                      duration_ms=duration,
                      details=str(quality_report))
            self.results["quality_report"] = quality_report
            self.results["df"] = df_clean
        except Exception as e:
            self._log("Data Cleaner", f"Error: {e}", status="error")
            df_clean = df.copy()
            self.results["quality_report"] = {"records_received": len(df), "valid_records": len(df)}
            self.results["df"] = df_clean

        # ── Step 4: Sentiment Analysis ────────────────────────────────────────
        self._notify("Running sentiment analysis...", 4, total_steps)
        t0 = time.time()
        try:
            df_clean = sentiment_tool.run(df_clean, feedback_col)
            duration = int((time.time() - t0) * 1000)
            sent_counts = df_clean["sentiment"].value_counts().to_dict()
            self._log("Sentiment Analyzer",
                      f"Sentiment analysis complete — "
                      f"Positive: {sent_counts.get('Positive', 0)}, "
                      f"Neutral: {sent_counts.get('Neutral', 0)}, "
                      f"Negative: {sent_counts.get('Negative', 0)}",
                      duration_ms=duration)
            self.results["df"] = df_clean
        except Exception as e:
            self._log("Sentiment Analyzer", f"Error: {e}", status="error")

        # ── Step 5: Emotion Analysis ──────────────────────────────────────────
        self._notify("Detecting customer emotions...", 5, total_steps)
        t0 = time.time()
        try:
            df_clean = emotion_tool.run(df_clean, feedback_col)
            duration = int((time.time() - t0) * 1000)
            emotion_counts = df_clean["emotion"].value_counts().to_dict()
            top_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else "N/A"
            self._log("Emotion Detector",
                      f"Emotions detected — dominant emotion: {top_emotion} "
                      f"({emotion_counts.get(top_emotion, 0)} occurrences)",
                      duration_ms=duration)
            self.results["df"] = df_clean
        except Exception as e:
            self._log("Emotion Detector", f"Error: {e}", status="error")

        # ── Step 6: Topic Detection ───────────────────────────────────────────
        self._notify("Identifying feedback topics...", 6, total_steps)
        t0 = time.time()
        try:
            df_clean = topic_tool.run(df_clean, feedback_col)
            duration = int((time.time() - t0) * 1000)
            topic_counts = df_clean["topic"].value_counts()
            n_topics = len(topic_counts)
            top_topic = topic_counts.index[0] if len(topic_counts) > 0 else "N/A"
            self._log("Topic Detector",
                      f"{n_topics} topics identified — most common: {top_topic}",
                      duration_ms=duration)
            self.results["df"] = df_clean
        except Exception as e:
            self._log("Topic Detector", f"Error: {e}", status="error")

        # ── Step 7: Issue Detection ───────────────────────────────────────────
        self._notify("Detecting complaints and issues...", 7, total_steps)
        t0 = time.time()
        try:
            df_clean = issue_detector.run(df_clean, feedback_col)
            duration = int((time.time() - t0) * 1000)
            complaint_count = df_clean["is_complaint"].sum()
            issue_summary = issue_detector.get_issue_summary(df_clean)
            self._log("Issue Detector",
                      f"{complaint_count:,} complaints detected across "
                      f"{len(issue_summary)} distinct issue types",
                      duration_ms=duration)
            self.results["df"] = df_clean
            self.results["issue_summary"] = issue_summary
        except Exception as e:
            self._log("Issue Detector", f"Error: {e}", status="error")
            self.results["issue_summary"] = pd.DataFrame()

        # ── Step 8: Recurring Issue Detection ────────────────────────────────
        self._notify("Clustering recurring issues...", 8, total_steps)
        t0 = time.time()
        try:
            recurring = recurring_issue_detector.run(df_clean, feedback_col)
            duration = int((time.time() - t0) * 1000)
            self._log("Recurring Issue Detector",
                      f"{len(recurring)} recurring issue clusters identified",
                      duration_ms=duration,
                      details="; ".join([r["issue_label"] for r in recurring[:5]]))
            self.results["recurring_issues"] = recurring
        except Exception as e:
            self._log("Recurring Issue Detector", f"Error: {e}", status="error")
            self.results["recurring_issues"] = []

        # ── Step 9: Priority Scoring ──────────────────────────────────────────
        self._notify("Calculating priority scores...", 9, total_steps)
        t0 = time.time()
        try:
            df_clean = priority_engine.run(df_clean, feedback_col)
            duration = int((time.time() - t0) * 1000)
            priority_dist = priority_engine.get_priority_distribution(df_clean)
            self._log("Priority Engine",
                      f"Priority calculated — HIGH: {priority_dist['HIGH']}, "
                      f"MEDIUM: {priority_dist['MEDIUM']}, LOW: {priority_dist['LOW']}",
                      duration_ms=duration)
            self.results["df"] = df_clean
            self.results["priority_distribution"] = priority_dist
        except Exception as e:
            self._log("Priority Engine", f"Error: {e}", status="error")
            self.results["priority_distribution"] = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}

        # ── Step 10: Trend Analysis ───────────────────────────────────────────
        self._notify("Analyzing trends...", 10, total_steps)
        t0 = time.time()
        try:
            trend_data = trend_analyzer.run(df_clean, date_col if has_date else None)
            duration = int((time.time() - t0) * 1000)
            if trend_data["date_available"]:
                self._log("Trend Analyzer",
                          "Time-based trend analysis complete — sentiment and volume trends calculated",
                          duration_ms=duration)
            else:
                self._log("Trend Analyzer",
                          "Trend analysis skipped — no date column available",
                          details="Provide a date column to enable trend analysis")
            self.results["trend_data"] = trend_data
        except Exception as e:
            self._log("Trend Analyzer", f"Error: {e}", status="error")
            self.results["trend_data"] = {"date_available": False}

        # ── Step 11: Insight Generation ───────────────────────────────────────
        self._notify("Generating business insights...", 11, total_steps)
        t0 = time.time()
        try:
            insights = insight_generator.generate_insights(
                df_clean,
                self.results.get("recurring_issues", []),
                self.results.get("quality_report", {}),
                self.results.get("trend_data", {}),
            )
            duration = int((time.time() - t0) * 1000)
            self._log("Insight Generator",
                      f"{len(insights)} business insights generated",
                      duration_ms=duration)
            self.results["insights"] = insights
        except Exception as e:
            self._log("Insight Generator", f"Error: {e}", status="error")
            self.results["insights"] = []

        # ── Step 12: Recommendation Generation ───────────────────────────────
        self._notify("Generating recommendations...", 12, total_steps)
        t0 = time.time()
        try:
            recommendations = recommendation_engine.generate_recommendations(
                self.results.get("recurring_issues", []),
                self.results.get("issue_summary"),
            )
            duration = int((time.time() - t0) * 1000)
            high_recs = sum(1 for r in recommendations if r["priority"] == "HIGH")
            self._log("Recommendation Engine",
                      f"{len(recommendations)} recommendations generated "
                      f"({high_recs} high-priority actions)",
                      duration_ms=duration)
            self.results["recommendations"] = recommendations
        except Exception as e:
            self._log("Recommendation Engine", f"Error: {e}", status="error")
            self.results["recommendations"] = []

        # ── Final ─────────────────────────────────────────────────────────────
        self._log("Agent", "Analysis complete — all results ready for review",
                  details=f"Total steps: {self._step_counter}")

        self.results["activity_log"] = [a.to_dict() for a in self.activity_log]
        self.results["df"] = df_clean

        # Store optional data
        self.results["date_col"] = date_col
        self.results["rating_col"] = rating_col
        self.results["feedback_col"] = feedback_col

        return self.results

    def analyze_single(self, text: str) -> Dict:
        """Analyze a single feedback string and return full analysis."""
        import pandas as pd

        row_df = pd.DataFrame({"feedback": [text]})

        # Run tools on single row
        result = {}

        # Sentiment
        sent = sentiment_tool.analyze_single(text)
        result.update(sent)

        # Emotion
        from tools.emotion_tool import detect_emotion
        emo = detect_emotion(text)
        result.update(emo)

        # Topic
        from tools.topic_tool import detect_topic
        topic, topic_conf = detect_topic(text)
        result["topic"] = topic
        result["topic_confidence"] = topic_conf

        # Issue
        from tools.issue_detector import detect_issue, is_complaint
        issue, issue_conf = detect_issue(text)
        result["detected_issue"] = issue
        result["issue_confidence"] = issue_conf
        result["is_complaint"] = is_complaint(text)

        # Priority
        from tools.priority_engine import score_priority
        priority_result = score_priority({
            "sentiment": result.get("sentiment"),
            "emotion": result.get("emotion"),
            "detected_issue": result.get("detected_issue"),
            "is_complaint": result.get("is_complaint"),
            "feedback": text,
        })
        result.update(priority_result)

        # Recommendation
        issue_label = result.get("detected_issue", "General Complaint")
        rec_template = recommendation_engine.ISSUE_RECOMMENDATIONS.get(
            issue_label, recommendation_engine.DEFAULT_RECOMMENDATION
        )
        result["recommended_action"] = rec_template["action"]
        result["recommendation_details"] = rec_template["details"]

        return result
