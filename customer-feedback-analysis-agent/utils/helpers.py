"""
Utility helpers for the Customer Feedback Analysis Agent.
"""

from __future__ import annotations

import pandas as pd
from typing import Dict, Optional

# Plotly is only needed by the local Streamlit UI (app.py). It is intentionally
# NOT installed in the Vercel serverless bundle (the web UI renders its charts
# with Plotly.js in the browser), so the import is optional here.
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:  # pragma: no cover - serverless path
    px = None
    go = None
    PLOTLY_AVAILABLE = False

# ── Color Palette (Light Wine & Cream) ─────────────────────────────────────────
COLORS = {
    "primary": "#F7F2EC",
    "secondary": "#FAF6F0",
    "accent": "#722F37",       # Classic Bordeaux Wine
    "wine_light": "#A3485E",   # Soft Rose Wine
    "wine_dark": "#4A1521",    # Deep Velvet Wine
    "positive": "#2D7A58",     # Botanical Sage Emerald
    "negative": "#9C1D3A",     # Bold Wine Crimson
    "neutral": "#7D656A",      # Dusty Rose Taupe
    "warning": "#BA6A24",      # Warm Amber Sienna
    "info": "#4A6FA5",         # Soft Slate Blue
    "HIGH": "#8B1430",         # Deep Wine High Priority
    "MEDIUM": "#BA6A24",       # Warm Sienna Medium Priority
    "LOW": "#2D7A58",          # Sage Low Priority
}

SENTIMENT_COLORS = {
    "Positive": "#2D7A58",
    "Neutral": "#7D656A",
    "Negative": "#9C1D3A",
}

EMOTION_COLORS = {
    "Happy": "#2D7A58",
    "Satisfied": "#459B73",
    "Neutral": "#7D656A",
    "Confused": "#5A7C99",
    "Disappointed": "#BA6A24",
    "Frustrated": "#B33951",
    "Angry": "#8B1430",
}

PRIORITY_COLORS = {
    "HIGH": "#8B1430",
    "MEDIUM": "#BA6A24",
    "LOW": "#2D7A58",
}


def detect_feedback_column(df: pd.DataFrame) -> Optional[str]:
    """Auto-detect the feedback text column."""
    candidates = [
        "feedback", "review", "comment", "customer_feedback",
        "review_text", "complaint", "text", "message", "description"
    ]
    for col in candidates:
        if col in df.columns:
            return col
    # Fallback: longest average text column
    text_cols = df.select_dtypes(include="object").columns.tolist()
    if text_cols:
        avg_lengths = {col: df[col].dropna().astype(str).str.len().mean()
                       for col in text_cols}
        return max(avg_lengths, key=avg_lengths.get)
    return None


def detect_date_column(df: pd.DataFrame) -> Optional[str]:
    """Auto-detect date column."""
    candidates = ["date", "created_at", "timestamp", "review_date", "feedback_date", "time"]
    for col in candidates:
        if col in df.columns:
            return col
    # Try to find any column that can be parsed as date
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            return col
    return None


def detect_rating_column(df: pd.DataFrame) -> Optional[str]:
    """Auto-detect rating column."""
    candidates = ["rating", "score", "stars", "review_score", "customer_rating"]
    for col in candidates:
        if col in df.columns:
            return col
    # Numeric columns with values 1-5
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in num_cols:
        vals = df[col].dropna()
        if len(vals) > 0 and vals.min() >= 1 and vals.max() <= 5:
            return col
    return None


# ── Chart builders ────────────────────────────────────────────────────────────

CHART_FONT = dict(family="Plus Jakarta Sans, Inter, system-ui, sans-serif", color="#5C4247", size=12)
CHART_TITLE_FONT = dict(family="Outfit, Playfair Display, system-ui, sans-serif", size=15, color="#380E16")

def sentiment_pie_chart(df: pd.DataFrame) -> Optional[go.Figure]:
    if "sentiment" not in df.columns:
        return None
    counts = df["sentiment"].value_counts().reset_index()
    counts.columns = ["Sentiment", "Count"]
    fig = px.pie(
        counts, names="Sentiment", values="Count",
        color="Sentiment", color_discrete_map=SENTIMENT_COLORS,
        title="Sentiment Distribution",
        hole=0.55,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=CHART_FONT,
        title_font=CHART_TITLE_FONT,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, font=CHART_FONT),
        margin=dict(t=50, b=50, l=20, r=20),
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        marker=dict(line=dict(color="#FFFFFF", width=2))
    )
    return fig


def emotion_bar_chart(df: pd.DataFrame) -> Optional[go.Figure]:
    if "emotion" not in df.columns:
        return None
    counts = df["emotion"].value_counts().reset_index()
    counts.columns = ["Emotion", "Count"]
    fig = px.bar(
        counts, x="Emotion", y="Count",
        color="Emotion", color_discrete_map=EMOTION_COLORS,
        title="Emotion Breakdown",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=CHART_FONT,
        title_font=CHART_TITLE_FONT,
        showlegend=False,
        margin=dict(t=50, b=50, l=20, r=20),
        xaxis=dict(gridcolor="rgba(114, 47, 55, 0.08)", tickfont=CHART_FONT),
        yaxis=dict(gridcolor="rgba(114, 47, 55, 0.08)", tickfont=CHART_FONT),
    )
    fig.update_traces(marker=dict(line=dict(color="#FFFFFF", width=1), opacity=0.9))
    return fig


def topic_bar_chart(df: pd.DataFrame) -> Optional[go.Figure]:
    if "topic" not in df.columns:
        return None
    counts = df["topic"].value_counts().reset_index()
    counts.columns = ["Topic", "Count"]
    fig = px.bar(
        counts, x="Count", y="Topic", orientation="h",
        title="Feedback Topics",
        color="Count",
        color_continuous_scale=[[0, "#E8D8D3"], [0.5, "#B34A62"], [1, "#6B1D2F"]],
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=CHART_FONT,
        title_font=CHART_TITLE_FONT,
        showlegend=False,
        margin=dict(t=50, b=50, l=150, r=20),
        yaxis=dict(gridcolor="rgba(114, 47, 55, 0.08)", tickfont=CHART_FONT),
        xaxis=dict(gridcolor="rgba(114, 47, 55, 0.08)", tickfont=CHART_FONT),
        coloraxis_showscale=False,
    )
    fig.update_traces(marker=dict(line=dict(color="#FFFFFF", width=1)))
    return fig


def priority_donut_chart(priority_dist: Dict) -> go.Figure:
    labels = list(priority_dist.keys())
    values = list(priority_dist.values())
    colors_list = [PRIORITY_COLORS.get(l, "#7D656A") for l in labels]
    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values,
        hole=0.55, marker=dict(colors=colors_list, line=dict(color="#FFFFFF", width=2)),
        textinfo="percent+label",
    )])
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=CHART_FONT,
        title="Priority Distribution",
        title_font=CHART_TITLE_FONT,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, font=CHART_FONT),
        margin=dict(t=50, b=50, l=20, r=20),
    )
    return fig


def sentiment_trend_chart(trend_df: pd.DataFrame) -> Optional[go.Figure]:
    if trend_df is None or len(trend_df) == 0:
        return None
    fig = px.line(
        trend_df, x="period", y="count", color="sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        title="Sentiment Trend Over Time",
        markers=True,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=CHART_FONT,
        title_font=CHART_TITLE_FONT,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, font=CHART_FONT),
        margin=dict(t=50, b=80, l=20, r=20),
        xaxis=dict(gridcolor="rgba(114, 47, 55, 0.08)", tickfont=CHART_FONT),
        yaxis=dict(gridcolor="rgba(114, 47, 55, 0.08)", tickfont=CHART_FONT),
    )
    return fig


def volume_trend_chart(trend_df: pd.DataFrame) -> Optional[go.Figure]:
    if trend_df is None or len(trend_df) == 0:
        return None
    fig = px.area(
        trend_df, x="period", y="count",
        title="Feedback Volume Ingestion",
        color_discrete_sequence=["#8B1E3F"],
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=CHART_FONT,
        title_font=CHART_TITLE_FONT,
        margin=dict(t=50, b=50, l=20, r=20),
        xaxis=dict(gridcolor="rgba(114, 47, 55, 0.08)", tickfont=CHART_FONT),
        yaxis=dict(gridcolor="rgba(114, 47, 55, 0.08)", tickfont=CHART_FONT),
    )
    return fig


def recurring_issues_chart(recurring_issues: list) -> Optional[go.Figure]:
    if not recurring_issues:
        return None
    labels = [r["issue_label"] for r in recurring_issues[:10]]
    values = [r["occurrences"] for r in recurring_issues[:10]]
    priorities = [r["priority"] for r in recurring_issues[:10]]
    colors_list = [PRIORITY_COLORS.get(p, "#7D656A") for p in priorities]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors_list,
        text=[f"{v} ({r['percentage']}%)" for v, r in
              zip(values, recurring_issues[:10])],
        textposition="outside",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=CHART_FONT,
        title="Top Recurring Semantic Clusters",
        title_font=CHART_TITLE_FONT,
        showlegend=False,
        margin=dict(t=50, b=50, l=200, r=80),
        yaxis=dict(gridcolor="rgba(114, 47, 55, 0.08)", autorange="reversed", tickfont=CHART_FONT),
        xaxis=dict(gridcolor="rgba(114, 47, 55, 0.08)", tickfont=CHART_FONT),
    )
    return fig
