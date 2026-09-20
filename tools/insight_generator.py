"""
Insight Generator Tool
======================
Generates data-driven insights from analysis results.
ALL insights are computed from actual data — nothing is hard-coded.
"""

import pandas as pd
from typing import List, Dict, Any


def generate_insights(
    df: pd.DataFrame,
    recurring_issues: List[Dict],
    quality_report: Dict,
    trend_data: Dict,
) -> List[Dict]:
    """
    Generate a list of insight dicts from the analysis.

    Each insight has: title, description, severity (info/warning/critical), icon
    """
    insights = []

    if df is None or len(df) == 0:
        return insights

    total = len(df)

    # ── Sentiment Insights ────────────────────────────────────────────────────
    if "sentiment" in df.columns:
        sentiment_counts = df["sentiment"].value_counts()
        negative_pct = round(sentiment_counts.get("Negative", 0) / total * 100, 1)
        positive_pct = round(sentiment_counts.get("Positive", 0) / total * 100, 1)

        if negative_pct >= 40:
            insights.append({
                "title": "High Negative Feedback Rate",
                "description": (
                    f"{negative_pct}% of customer feedback is negative — "
                    "this significantly exceeds the healthy threshold of 20%. "
                    "Immediate review of the main complaint areas is recommended."
                ),
                "severity": "critical",
                "icon": "🔴",
            })
        elif negative_pct >= 25:
            insights.append({
                "title": "Elevated Negative Feedback",
                "description": (
                    f"{negative_pct}% of customer feedback is negative. "
                    "This is above the ideal range and warrants attention."
                ),
                "severity": "warning",
                "icon": "🟡",
            })

        if positive_pct >= 60:
            insights.append({
                "title": "Strong Positive Customer Sentiment",
                "description": (
                    f"{positive_pct}% of feedback is positive, indicating high "
                    "customer satisfaction. Focus on maintaining current standards."
                ),
                "severity": "info",
                "icon": "🟢",
            })

    # ── Topic Insights ────────────────────────────────────────────────────────
    if "topic" in df.columns:
        topic_counts = df["topic"].value_counts()
        top_topic = topic_counts.index[0]
        top_topic_pct = round(topic_counts.iloc[0] / total * 100, 1)

        insights.append({
            "title": f"Top Feedback Topic: {top_topic}",
            "description": (
                f"{top_topic_pct}% of all feedback is related to '{top_topic}'. "
                "This is the primary area of customer concern and should be prioritized."
            ),
            "severity": "warning" if top_topic_pct > 30 else "info",
            "icon": "📌",
        })

        # Cross sentiment x topic
        if "sentiment" in df.columns:
            neg_by_topic = (
                df[df["sentiment"] == "Negative"]
                .groupby("topic")
                .size()
                .sort_values(ascending=False)
            )
            if len(neg_by_topic) > 0:
                worst_topic = neg_by_topic.index[0]
                worst_count = neg_by_topic.iloc[0]
                insights.append({
                    "title": f"Most Negative Topic: {worst_topic}",
                    "description": (
                        f"'{worst_topic}' has the highest concentration of negative feedback "
                        f"with {worst_count} negative reviews. This represents a critical pain point."
                    ),
                    "severity": "critical" if worst_count >= 20 else "warning",
                    "icon": "⚠️",
                })

    # ── Priority Insights ─────────────────────────────────────────────────────
    if "priority" in df.columns:
        high_priority = (df["priority"] == "HIGH").sum()
        high_pct = round(high_priority / total * 100, 1)

        if high_priority > 0:
            insights.append({
                "title": f"{high_priority} High-Priority Issues Detected",
                "description": (
                    f"{high_pct}% of feedback ({high_priority} records) has been classified as "
                    "HIGH priority based on sentiment, emotion, complaint type, and urgency signals. "
                    "These require immediate business action."
                ),
                "severity": "critical" if high_pct > 20 else "warning",
                "icon": "🚨",
            })

    # ── Recurring Issue Insights ──────────────────────────────────────────────
    if recurring_issues:
        top_issue = recurring_issues[0]
        insights.append({
            "title": f"Most Frequent Recurring Issue: {top_issue['issue_label']}",
            "description": (
                f"'{top_issue['issue_label']}' appears {top_issue['occurrences']} times "
                f"({top_issue['percentage']}% of all feedback), making it the most critical "
                "recurring problem. This pattern suggests a systemic operational issue."
            ),
            "severity": "critical" if top_issue["priority"] == "HIGH" else "warning",
            "icon": "🔁",
        })

        high_recurring = [r for r in recurring_issues if r["priority"] == "HIGH"]
        if len(high_recurring) >= 3:
            insights.append({
                "title": f"{len(high_recurring)} Systemic High-Priority Issues Found",
                "description": (
                    f"Analysis detected {len(high_recurring)} distinct recurring issues at HIGH priority, "
                    "suggesting multiple systemic operational failures that need strategic intervention."
                ),
                "severity": "critical",
                "icon": "📊",
            })

    # ── Emotion Insights ──────────────────────────────────────────────────────
    if "emotion" in df.columns:
        emotion_counts = df["emotion"].value_counts()
        top_emotion = emotion_counts.index[0]
        top_emotion_pct = round(emotion_counts.iloc[0] / total * 100, 1)

        if top_emotion in ("Angry", "Frustrated", "Disappointed"):
            insights.append({
                "title": f"Dominant Customer Emotion: {top_emotion}",
                "description": (
                    f"{top_emotion_pct}% of customers express '{top_emotion}' emotion in their feedback. "
                    "High levels of this emotion indicate a poor customer experience that needs addressing."
                ),
                "severity": "warning",
                "icon": "😤",
            })

    # ── Complaint Rate Insight ─────────────────────────────────────────────────
    if "is_complaint" in df.columns:
        complaint_count = df["is_complaint"].sum()
        complaint_pct = round(complaint_count / total * 100, 1)
        insights.append({
            "title": f"Complaint Rate: {complaint_pct}%",
            "description": (
                f"{complaint_count} out of {total} feedback records contain identifiable complaints. "
                f"A complaint rate of {complaint_pct}% requires systematic response management."
            ),
            "severity": "critical" if complaint_pct > 40 else "warning" if complaint_pct > 20 else "info",
            "icon": "📢",
        })

    # ── Trend Insights ────────────────────────────────────────────────────────
    if trend_data.get("date_available") and trend_data.get("sentiment_trend") is not None:
        st = trend_data["sentiment_trend"]
        neg_trend = st[st["sentiment"] == "Negative"].sort_values("period")
        if len(neg_trend) >= 2:
            first_half = neg_trend.iloc[: len(neg_trend) // 2]["count"].mean()
            second_half = neg_trend.iloc[len(neg_trend) // 2 :]["count"].mean()
            if second_half > first_half * 1.2:
                insights.append({
                    "title": "Increasing Negative Sentiment Trend",
                    "description": (
                        "Negative feedback volume has increased over time. "
                        "The trend shows growing customer dissatisfaction that requires urgent attention."
                    ),
                    "severity": "critical",
                    "icon": "📈",
                })

    # Deduplicate by title
    seen = set()
    unique_insights = []
    for ins in insights:
        if ins["title"] not in seen:
            seen.add(ins["title"])
            unique_insights.append(ins)

    return unique_insights
