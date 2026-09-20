# Customer Feedback Analysis Agent — Architecture

## Overview

The Customer Feedback Analysis Agent is an autonomous AI system built on an orchestrator-tool pattern. A single main agent coordinates multiple specialized NLP/ML tools, each responsible for one analytical task.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER (Streamlit UI)                          │
│                                                                 │
│  ┌──────────────┐          ┌──────────────┐                    │
│  │ Single Text  │          │ CSV/Excel    │                    │
│  │ Input        │          │ Upload       │                    │
│  └──────┬───────┘          └──────┬───────┘                    │
└─────────┼────────────────────────┼────────────────────────────-┘
          │                        │
          ▼                        ▼
┌─────────────────────────────────────────────────────────────────┐
│              CustomerFeedbackAgent (Orchestrator)               │
│                                                                 │
│  • Observes input                                               │
│  • Selects required tools                                       │
│  • Calls tools in sequence                                      │
│  • Collects and aggregates results                              │
│  • Maintains activity log                                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
           ┌─────────────────┴──────────────────────┐
           │            Tool Calls                   │
           ▼                                         ▼
┌──────────────────────────────────────────────────────────────────┐
│                     SPECIALIZED TOOLS                            │
│                                                                  │
│  1. Data Cleaner         → Normalizes and validates text         │
│  2. Sentiment Analyzer   → Positive / Neutral / Negative         │
│  3. Emotion Detector     → Happy / Frustrated / Angry / etc.     │
│  4. Topic Detector       → Delivery / Support / Payment / etc.   │
│  5. Issue Detector       → Specific complaint identification      │
│  6. Recurring Issue Det. → TF-IDF + DBSCAN clustering            │
│  7. Priority Engine      → Multi-factor priority scoring         │
│  8. Trend Analyzer       → Time-series analysis                  │
│  9. Insight Generator    → Data-driven insight creation          │
│  10. Recommendation Eng. → Evidence-based recommendations        │
│  11. Report Generator    → PDF + CSV export                      │
└──────────────────────────────────────────────────────────────────┘
```

## Agent Workflow

```
Input Received
    │
    ▼
Understand Data Structure
    │
    ▼
Data Cleaning Tool
    │
    ▼
Sentiment Analysis Tool (VADER/TextBlob)
    │
    ▼
Emotion Analysis Tool (Keyword/Rule-based)
    │
    ▼
Topic Detection Tool (Keyword + TF-IDF)
    │
    ▼
Issue Detector Tool (Regex pattern matching)
    │
    ▼
Recurring Issue Detector (TF-IDF + DBSCAN)
    │
    ▼
Priority Engine (Multi-factor scoring)
    │
    ▼
Trend Analyzer (Time-series if date available)
    │
    ▼
Insight Generator (Rule-based from data)
    │
    ▼
Recommendation Engine (Issue → Action mapping)
    │
    ▼
Report Generator (PDF + CSV)
    │
    ▼
Results Displayed in Dashboard
```

## Technology Choices

| Component | Technology | Reason |
|-----------|-----------|--------|
| UI | Streamlit | Rapid Python-based dashboard, no JS needed |
| Sentiment | VADER + TextBlob | No API needed, robust for review text |
| Topic Detection | TF-IDF + Keywords | Efficient, no training data needed |
| Clustering | DBSCAN (sklearn) | Density-based, handles noise, no cluster count needed |
| PDF | ReportLab | Full control over professional layout |
| Data | Pandas | Industry standard, fast |
| Charts | Plotly | Interactive, beautiful |

## Key Design Decisions

1. **No paid API required** — All NLP runs locally with NLTK/TextBlob/sklearn
2. **Fallback chain** — Every tool has a fallback if a library is unavailable
3. **Transparent activity** — Full agent activity log shown to user
4. **Evidence-based** — Every insight and recommendation derived from actual data
5. **Modular architecture** — Each tool can be replaced/upgraded independently
