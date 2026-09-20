# 🤖 Customer Feedback Analysis Agent

> **An autonomous AI agent that transforms customer feedback into actionable business intelligence.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🌐 Live Deployment (Vercel)

This project ships **two interfaces over one shared agent pipeline**:

| Interface | Entry point | Run with |
|-----------|-------------|----------|
| Web app (production / Vercel) | `api/index.py` (Flask) | `python api/index.py` → http://127.0.0.1:5000 |
| Streamlit app (local) | `app.py` | `streamlit run app.py` → http://localhost:8501 |

```bash
# Web version — same code that runs on Vercel
pip install -r requirements.txt
python api/index.py

# Streamlit version — full local feature set
pip install -r requirements-local.txt
streamlit run app.py
```

👉 **Full deployment steps: [DEPLOYMENT.md](DEPLOYMENT.md)**

Optional environment variables (set in Vercel → Settings → Environment Variables,
never in code): `GROQ_API_KEY`, `GROQ_MODEL`, `MAX_ROWS`.


---

## 📋 Problem Statement

Businesses receive thousands of customer reviews and complaints daily but lack the tools to:
- Automatically detect specific issues in feedback
- Identify recurring complaint patterns at scale
- Prioritize which problems need immediate attention
- Generate actionable recommendations with evidence

## 🎯 Objectives

1. Build an autonomous AI agent for customer feedback analysis
2. Implement multi-tool NLP/ML pipeline (sentiment, emotion, topic, issue detection)
3. Detect recurring issues using unsupervised semantic clustering
4. Generate data-driven insights and evidence-based recommendations
5. Produce professional reports (PDF + CSV)

---

## 🚀 Features

| Feature | Description |
|---------|-------------|
| 🧠 Sentiment Analysis | Classify feedback as Positive/Neutral/Negative with confidence scores |
| ❤️ Emotion Detection | Detect 7 emotions: Happy, Satisfied, Neutral, Confused, Frustrated, Angry, Disappointed |
| 🏷️ Topic Classification | Auto-classify into 9 business topics (Delivery, Support, Payment, etc.) |
| ⚠️ Issue Detection | Identify 14 specific complaint types using pattern matching |
| 🔁 Recurring Issues | Cluster similar complaints using TF-IDF + DBSCAN |
| 🚨 Priority Engine | Multi-factor scoring (sentiment + emotion + frequency + urgency) |
| 💡 AI Insights | Dynamically generated insights from actual data |
| 🎯 Recommendations | Evidence-based actionable business recommendations |
| 📄 PDF Reports | Professional downloadable reports |
| 📊 Interactive Charts | Plotly-powered interactive visualizations |
| 🤖 Agent Activity Log | Full transparent workflow log |

---

## 🏗️ Agentic Architecture

### Traditional Approach:
```
Input → ML Model → Output
```

### Our Approach:
```
Input
  ↓
Customer Feedback Analysis Agent (Orchestrator)
  ↓                    ↓                    ↓
Data Cleaner     Sentiment Tool      Emotion Tool
  ↓                    ↓                    ↓
Topic Tool       Issue Detector    Recurring Issue Detector
  ↓                    ↓                    ↓
Priority Engine  Trend Analyzer    Insight Generator
  ↓                    ↓                    ↓
Recommendation Engine          Report Generator
  ↓
Complete Business Intelligence Report
```

The agent **autonomously orchestrates** all tools, selects applicable ones based on available data, and synthesizes results into actionable outputs.

---

## 🔄 Workflow

```
1. User uploads CSV/Excel or types feedback
2. Agent observes and understands input structure
3. Data Cleaner normalizes and validates feedback
4. Sentiment Analyzer classifies each feedback (VADER/TextBlob)
5. Emotion Detector identifies customer emotional state
6. Topic Detector classifies feedback into business categories
7. Issue Detector identifies specific complaints via regex patterns
8. Recurring Issue Detector clusters similar complaints (TF-IDF + DBSCAN)
9. Priority Engine scores urgency (multi-factor weighted scoring)
10. Trend Analyzer processes time-series data (if date available)
11. Insight Generator creates data-driven business insights
12. Recommendation Engine produces actionable recommendations
13. Report Generator creates PDF + CSV exports
14. Dashboard displays all results with interactive charts
```

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| UI/Dashboard | Streamlit |
| Data Processing | Pandas, NumPy |
| Sentiment Analysis | NLTK VADER, TextBlob |
| ML/Clustering | Scikit-learn (TF-IDF, DBSCAN) |
| Visualization | Plotly |
| PDF Generation | ReportLab |
| Excel Support | OpenPyXL |
| Language | Python 3.10+ |

> ✅ **No paid API required** — all models run locally

---

## 📦 Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/customer-feedback-analysis-agent.git
cd customer-feedback-analysis-agent

# 2. Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK data (auto-downloads on first run, or run manually)
python -c "import nltk; nltk.download('vader_lexicon')"
```

---

## ▶️ How to Run

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📂 Dataset Format

### Required
| Column | Description |
|--------|-------------|
| `feedback` / `review` / `comment` | Customer feedback text (required) |

### Optional (auto-detected)
| Column | Description |
|--------|-------------|
| `date` / `created_at` | Date of feedback (enables trend analysis) |
| `rating` / `score` | Star rating 1-5 (enables rating analysis) |
| `customer_id` | Customer identifier |
| `product` | Product name |
| `channel` | Feedback channel |

---

## 💬 Example Input

```
"The product arrived 6 days late. When I contacted customer support, 
they did not respond for 3 days. Very disappointed with the experience."
```

## 📊 Example Output

```
Sentiment:    NEGATIVE (Confidence: 91%)
Emotion:      Frustrated
Topic:        Delivery / Customer Support
Issue:        Delivery Delay + Slow Customer Support
Priority:     HIGH

Priority Reasons:
• Negative sentiment detected
• Strong negative emotion (Frustrated)
• Complaint detected
• High-impact issue type (Delivery Delay)

Recommended Action:
Review logistics partner performance and introduce monitoring for delayed shipments.
Review support response SLA and increase support capacity during peak periods.
```

---

## 🗂️ Project Structure

```
customer-feedback-analysis-agent/
├── app.py                          # Main Streamlit application
├── requirements.txt
├── README.md
├── .gitignore
│
├── agents/
│   └── customer_feedback_agent.py  # Main orchestration agent
│
├── tools/
│   ├── data_cleaner.py             # Data normalization & validation
│   ├── sentiment_tool.py           # VADER/TextBlob sentiment analysis
│   ├── emotion_tool.py             # Keyword-based emotion detection
│   ├── topic_tool.py               # Topic classification
│   ├── issue_detector.py           # Issue/complaint detection
│   ├── recurring_issue_detector.py # TF-IDF + DBSCAN clustering
│   ├── priority_engine.py          # Multi-factor priority scoring
│   ├── trend_analyzer.py           # Time-series trend analysis
│   ├── insight_generator.py        # Data-driven insight generation
│   ├── recommendation_engine.py    # Evidence-based recommendations
│   └── report_generator.py         # PDF + CSV generation
│
├── data/
│   └── sample_feedback.csv         # 350 realistic sample records
│
├── utils/
│   └── helpers.py                  # Chart builders, column detection
│
├── tests/
│   ├── test_cleaning.py
│   ├── test_sentiment.py
│   ├── test_priority.py
│   └── test_issues.py
│
└── docs/
    ├── architecture.md
    ├── REPORT_CONTENT.md
    └── VIVA.md
```

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_cleaning.py -v
pytest tests/test_sentiment.py -v
pytest tests/test_priority.py -v
pytest tests/test_issues.py -v
```

---

## ⚙️ Limitations

- English language only (currently)
- Sarcasm detection not fully implemented
- Context limited to individual feedback records
- Clustering accuracy depends on dataset size (better with 100+ records)

---

## 🔭 Future Scope

1. **LLM Integration**: Add Gemini/GPT-4 with local fallback for richer insights
2. **Multilingual**: Translation layer for non-English feedback
3. **Real-time**: Connect to live CRM/social media APIs
4. **Automated Alerts**: Email/Slack notifications for HIGH priority issues
5. **Fine-tuned Models**: Domain-specific BERT models for better accuracy
6. **REST API**: Expose as API for CRM integration

---

## 🎓 How This Qualifies as Agentic AI

1. **Autonomous Operation**: Agent runs the complete pipeline without human step-by-step guidance
2. **Tool Orchestration**: Main agent coordinates 11 specialized tools
3. **Intelligent Decision-Making**: Agent selects tools based on available data (trend analysis only if date exists)
4. **Observation → Planning → Action**: Visible in the Activity Log page
5. **Multi-step Reasoning**: Results from early tools inform later tools (sentiment → priority scoring)
6. **Transparent**: Full activity log shows exactly what the agent did and why
7. **Goal-Directed**: Agent's goal is clear (generate actionable business intelligence) and it autonomously achieves it

---

## 📝 Conclusion

The Customer Feedback Analysis Agent demonstrates a complete, production-quality agentic AI system that transforms raw customer feedback into structured business intelligence. It goes far beyond sentiment analysis to provide issue detection, recurring pattern discovery, priority scoring, and actionable recommendations — all autonomously, without paid APIs, and with full transparency.

---

*B.Tech Mini Project — Agentic AI & Automation (Flexi Credit Course)*
