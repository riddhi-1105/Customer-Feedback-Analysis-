"""
Customer Feedback Analysis Agent — Vercel / WSGI Entry Point
=============================================================

This module exposes a top-level ``app`` object (a Flask WSGI application),
which is what the Vercel Python runtime looks for. It reuses the EXACT same
agent + tools pipeline used by the Streamlit app (``app.py``) — nothing about
the analysis logic is duplicated or re-implemented here.

Local run:   python api/index.py      →  http://127.0.0.1:5000
Vercel:      handled automatically via vercel.json
"""

import io
import os
import sys
import json
import logging
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
# On Vercel the function's working directory is not guaranteed, so we always
# resolve the project root relative to THIS file and put it on sys.path. This is
# what makes `from agents...` / `from tools...` imports work in a serverless env.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from flask import Flask, request, jsonify, send_file, render_template

from agents.customer_feedback_agent import CustomerFeedbackAgent
from utils.helpers import (
    detect_feedback_column, detect_date_column, detect_rating_column,
    SENTIMENT_COLORS, EMOTION_COLORS, PRIORITY_COLORS,
)
from tools import report_generator
from tools.groq_client import (
    is_groq_available, explain_feedback_with_groq,
    generate_llm_executive_summary, DEFAULT_MODEL,
)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("feedback-agent")

# ── Limits (serverless-safe) ──────────────────────────────────────────────────
# Vercel caps a request body at ~4.5 MB and a serverless function at 10s (Hobby).
MAX_UPLOAD_BYTES = 4 * 1024 * 1024        # 4 MB upload ceiling
MAX_ROWS = int(os.getenv("MAX_ROWS", "1500"))  # rows analysed per request

SAMPLE_CSV = ROOT / "data" / "sample_feedback.csv"

# ── Flask app — THIS is the top-level `app` Vercel requires ───────────────────
app = Flask(
    __name__,
    template_folder=str(ROOT / "web" / "templates"),
    static_folder=str(ROOT / "web" / "static"),
    static_url_path="/static",
)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

# Aliases so any Vercel runtime probe finds an entry point.
application = app
handler = app


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════
def _jsonable(obj):
    """Recursively convert numpy/pandas scalars into plain JSON types."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if obj is None:
        return None
    if isinstance(obj, (str, bool, int, float)):
        return obj
    # numpy / pandas scalars
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if pd.isna(obj) if not hasattr(obj, "__len__") else False:
        return None
    return str(obj)


def _df_to_records(df: pd.DataFrame) -> list:
    """Serialize the analysed DataFrame safely (drops non-JSON columns)."""
    safe = df.copy()
    if "priority_reasons" in safe.columns:
        safe["priority_reasons"] = safe["priority_reasons"].apply(
            lambda v: list(v) if isinstance(v, (list, tuple)) else ([] if v is None else [str(v)])
        )
    for col in safe.columns:
        if str(safe[col].dtype).startswith("datetime"):
            safe[col] = safe[col].dt.strftime("%Y-%m-%d")
    safe = safe.where(pd.notna(safe), None)
    return _jsonable(safe.to_dict(orient="records"))


def _trend_to_json(trend_data: dict) -> dict:
    """Convert the trend DataFrames inside trend_data to plain records."""
    out = {"date_available": bool(trend_data.get("date_available"))}
    for key in ("sentiment_trend", "volume_trend", "topic_trend", "issue_trend"):
        val = trend_data.get(key)
        if isinstance(val, pd.DataFrame) and not val.empty:
            tmp = val.copy()
            for col in tmp.columns:
                if str(tmp[col].dtype).startswith("datetime"):
                    tmp[col] = tmp[col].dt.strftime("%Y-%m-%d")
            out[key] = _jsonable(tmp.to_dict(orient="records"))
        elif isinstance(val, list):
            out[key] = _jsonable(val)
        else:
            out[key] = []
    return out


def _build_payload(results: dict, df_raw_cols: list) -> dict:
    """Shape the agent's results dict into the JSON the front-end consumes."""
    df = results.get("df", pd.DataFrame())
    issue_summary = results.get("issue_summary")
    if isinstance(issue_summary, pd.DataFrame):
        issue_summary = _jsonable(issue_summary.to_dict(orient="records"))
    else:
        issue_summary = []

    total = len(df)
    sent_counts = df["sentiment"].value_counts().to_dict() if "sentiment" in df else {}
    emo_counts = df["emotion"].value_counts().to_dict() if "emotion" in df else {}
    topic_counts = df["topic"].value_counts().to_dict() if "topic" in df else {}
    complaints = int(df["is_complaint"].sum()) if "is_complaint" in df else 0

    def pct(n):
        return round((n / total) * 100, 1) if total else 0.0

    metrics = {
        "total": total,
        "positive": int(sent_counts.get("Positive", 0)),
        "neutral": int(sent_counts.get("Neutral", 0)),
        "negative": int(sent_counts.get("Negative", 0)),
        "positive_pct": pct(sent_counts.get("Positive", 0)),
        "neutral_pct": pct(sent_counts.get("Neutral", 0)),
        "negative_pct": pct(sent_counts.get("Negative", 0)),
        "complaints": complaints,
        "complaint_pct": pct(complaints),
        "avg_priority_score": round(float(df["priority_score"].mean()), 1)
        if "priority_score" in df and total else 0.0,
    }

    return {
        "ok": True,
        "metrics": metrics,
        "quality_report": _jsonable(results.get("quality_report", {})),
        "priority_distribution": _jsonable(results.get("priority_distribution", {})),
        "sentiment_counts": _jsonable(sent_counts),
        "emotion_counts": _jsonable(emo_counts),
        "topic_counts": _jsonable(topic_counts),
        "issue_summary": issue_summary,
        "recurring_issues": _jsonable(results.get("recurring_issues", [])),
        "insights": _jsonable(results.get("insights", [])),
        "recommendations": _jsonable(results.get("recommendations", [])),
        "activity_log": _jsonable(results.get("activity_log", [])),
        "trend_data": _trend_to_json(results.get("trend_data", {})),
        "columns": {
            "feedback": results.get("feedback_col"),
            "date": results.get("date_col"),
            "rating": results.get("rating_col"),
            "all": list(df_raw_cols),
        },
        "rows": _df_to_records(df),
        "palette": {
            "sentiment": SENTIMENT_COLORS,
            "emotion": EMOTION_COLORS,
            "priority": PRIORITY_COLORS,
        },
    }


def _read_uploaded_csv(file_storage) -> pd.DataFrame:
    raw = file_storage.read()
    name = (file_storage.filename or "").lower()
    buf = io.BytesIO(raw)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(buf)
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            buf.seek(0)
            return pd.read_csv(buf, encoding=enc)
        except UnicodeDecodeError:
            continue
        except Exception:
            break
    buf.seek(0)
    return pd.read_csv(buf, encoding="latin-1", on_bad_lines="skip")


def _rows_to_df(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows or [])
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Routes — UI
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({"ok": True, "service": "customer-feedback-analysis-agent"})


@app.route("/api/status")
def status():
    """Tells the UI whether the optional Groq LLM layer is configured."""
    return jsonify({
        "ok": True,
        "groq_available": bool(is_groq_available()),
        "groq_model": DEFAULT_MODEL,
        "sample_available": SAMPLE_CSV.exists(),
        "max_rows": MAX_ROWS,
    })


# ══════════════════════════════════════════════════════════════════════════════
# Routes — Analysis
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Run the full 12-step agent pipeline.

    Accepts either:
      - multipart/form-data with a `file` field (CSV/XLSX), or
      - JSON  {"sample": true}  to analyse the bundled sample dataset.
    """
    try:
        df_raw = None
        feedback_col = date_col = rating_col = None

        if request.files.get("file"):
            df_raw = _read_uploaded_csv(request.files["file"])
            form = request.form
            feedback_col = form.get("feedback_col") or None
            date_col = form.get("date_col") or None
            rating_col = form.get("rating_col") or None
        else:
            body = request.get_json(silent=True) or {}
            if body.get("sample"):
                if not SAMPLE_CSV.exists():
                    return jsonify({"ok": False, "error": "Sample dataset not found."}), 404
                df_raw = pd.read_csv(SAMPLE_CSV)
            elif body.get("rows"):
                df_raw = pd.DataFrame(body["rows"])
            feedback_col = body.get("feedback_col") or None
            date_col = body.get("date_col") or None
            rating_col = body.get("rating_col") or None

        if df_raw is None or df_raw.empty:
            return jsonify({"ok": False, "error": "No data provided. Upload a CSV or load the sample dataset."}), 400

        raw_cols = df_raw.columns.tolist()

        # Auto-detect columns when the caller did not specify them.
        feedback_col = feedback_col if feedback_col in raw_cols else detect_feedback_column(df_raw)
        if not feedback_col:
            return jsonify({"ok": False, "error": "Could not find a text/feedback column in this file."}), 400
        date_col = date_col if date_col in raw_cols else detect_date_column(df_raw)
        rating_col = rating_col if rating_col in raw_cols else detect_rating_column(df_raw)

        truncated = False
        if len(df_raw) > MAX_ROWS:
            df_raw = df_raw.head(MAX_ROWS)
            truncated = True

        agent = CustomerFeedbackAgent()
        results = agent.analyze_dataset(df_raw, feedback_col, date_col, rating_col)

        payload = _build_payload(results, raw_cols)
        payload["truncated"] = truncated
        payload["max_rows"] = MAX_ROWS
        return jsonify(payload)

    except Exception as exc:  # noqa: BLE001
        logger.exception("Analysis failed")
        return jsonify({"ok": False, "error": f"Analysis failed: {exc}"}), 500


@app.route("/api/analyze-single", methods=["POST"])
def analyze_single():
    """Analyse one free-text feedback string."""
    try:
        body = request.get_json(silent=True) or {}
        text = (body.get("text") or "").strip()
        if not text:
            return jsonify({"ok": False, "error": "Please enter some feedback text."}), 400

        agent = CustomerFeedbackAgent()
        result = agent.analyze_single(text)
        return jsonify({"ok": True, "result": _jsonable(result), "text": text})
    except Exception as exc:  # noqa: BLE001
        logger.exception("Single analysis failed")
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/explain", methods=["POST"])
def explain():
    """Optional Groq LLM deep-dive for a single feedback item."""
    body = request.get_json(silent=True) or {}
    if not is_groq_available():
        return jsonify({"ok": False, "error": "Groq LLM is not configured. Set GROQ_API_KEY in your Vercel environment variables."}), 400
    try:
        text = explain_feedback_with_groq(
            feedback_text=body.get("text", ""),
            sentiment=body.get("sentiment", "N/A"),
            topic=body.get("topic", "N/A"),
            issue=body.get("issue", "N/A"),
            priority=body.get("priority", "N/A"),
        )
        if not text:
            return jsonify({"ok": False, "error": "The LLM did not return a response. Rule-based analysis above is still valid."}), 502
        return jsonify({"ok": True, "explanation": text, "model": DEFAULT_MODEL})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/executive-summary", methods=["POST"])
def executive_summary():
    """Optional Groq LLM executive debrief for the whole dataset."""
    body = request.get_json(silent=True) or {}
    if not is_groq_available():
        return jsonify({"ok": False, "error": "Groq LLM is not configured. Set GROQ_API_KEY in your Vercel environment variables."}), 400
    try:
        summary = generate_llm_executive_summary(
            metrics=body.get("metrics", {}),
            top_issues=body.get("top_issues", []),
        )
        if not summary:
            return jsonify({"ok": False, "error": "The LLM did not return a summary."}), 502
        return jsonify({"ok": True, "summary": summary, "model": DEFAULT_MODEL})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": str(exc)}), 500


# ══════════════════════════════════════════════════════════════════════════════
# Routes — Reports
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/report/csv", methods=["POST"])
def report_csv():
    """Export the analysed dataset as CSV."""
    try:
        body = request.get_json(silent=True) or {}
        df = _rows_to_df(body.get("rows"))
        if df.empty:
            return jsonify({"ok": False, "error": "No analysed data to export."}), 400
        csv_text = report_generator.generate_csv(df)
        return send_file(
            io.BytesIO(csv_text.encode("utf-8")),
            mimetype="text/csv",
            as_attachment=True,
            download_name="analyzed_feedback.csv",
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("CSV export failed")
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/report/pdf", methods=["POST"])
def report_pdf():
    """Export the full PDF report (same generator as the Streamlit app)."""
    try:
        body = request.get_json(silent=True) or {}
        df = _rows_to_df(body.get("rows"))
        if df.empty:
            return jsonify({"ok": False, "error": "No analysed data to export."}), 400

        pdf_bytes = report_generator.generate_pdf(
            df=df,
            quality_report=body.get("quality_report", {}) or {},
            recurring_issues=body.get("recurring_issues", []) or [],
            insights=body.get("insights", []) or [],
            recommendations=body.get("recommendations", []) or [],
            trend_data=body.get("trend_data", {}) or {},
        )
        if not pdf_bytes:
            return jsonify({"ok": False, "error": "PDF generation is unavailable (ReportLab missing). Use CSV export instead."}), 503

        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype="application/pdf",
            as_attachment=True,
            download_name="feedback_analysis_report.pdf",
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("PDF export failed")
        return jsonify({"ok": False, "error": str(exc)}), 500


# ══════════════════════════════════════════════════════════════════════════════
# Error handlers — always return JSON for /api/*, never an HTML stack trace
# ══════════════════════════════════════════════════════════════════════════════
@app.errorhandler(413)
def too_large(_):
    return jsonify({"ok": False, "error": "File too large. Vercel limits uploads to about 4 MB — try a smaller CSV."}), 413


@app.errorhandler(404)
def not_found(_):
    if request.path.startswith("/api/"):
        return jsonify({"ok": False, "error": "Endpoint not found."}), 404
    return render_template("index.html"), 200


@app.errorhandler(500)
def server_error(_):
    return jsonify({"ok": False, "error": "Internal server error."}), 500


# ── Local development server ──────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
