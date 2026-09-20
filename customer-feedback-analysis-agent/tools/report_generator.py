"""
Report Generator Tool
=====================
Generates a downloadable PDF report from the full analysis.
Uses ReportLab for PDF generation.
"""

import io
import datetime
from typing import List, Dict, Optional
import pandas as pd

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, PageBreak
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    _REPORTLAB_AVAILABLE = True
except ImportError:
    _REPORTLAB_AVAILABLE = False


def _build_pdf(
    df: pd.DataFrame,
    quality_report: Dict,
    recurring_issues: List[Dict],
    insights: List[Dict],
    recommendations: List[Dict],
    trend_data: Dict,
) -> bytes:
    """Build and return PDF bytes."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm
    )
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "CustomTitle", parent=styles["Title"],
        fontSize=22, textColor=colors.HexColor("#1a1a2e"),
        spaceAfter=6, alignment=TA_CENTER
    )
    heading1 = ParagraphStyle(
        "H1", parent=styles["Heading1"],
        fontSize=14, textColor=colors.HexColor("#16213e"),
        spaceBefore=16, spaceAfter=6,
        borderPad=4
    )
    heading2 = ParagraphStyle(
        "H2", parent=styles["Heading2"],
        fontSize=12, textColor=colors.HexColor("#0f3460"),
        spaceBefore=10, spaceAfter=4
    )
    body = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=10, textColor=colors.HexColor("#333333"),
        spaceAfter=4, leading=14, alignment=TA_JUSTIFY
    )
    accent = ParagraphStyle(
        "Accent", parent=styles["Normal"],
        fontSize=10, textColor=colors.HexColor("#e94560"),
        spaceAfter=4, fontName="Helvetica-Bold"
    )

    content = []
    now = datetime.datetime.now().strftime("%B %d, %Y %H:%M")

    # ── Cover ─────────────────────────────────────────────────────────────────
    content.append(Spacer(1, 2 * cm))
    content.append(Paragraph("Customer Feedback Analysis Agent", title_style))
    content.append(Paragraph("AI-Powered Feedback Intelligence Report", styles["Heading2"]))
    content.append(Spacer(1, 0.5 * cm))
    content.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#e94560")))
    content.append(Spacer(1, 0.3 * cm))
    content.append(Paragraph(f"Generated: {now}", styles["Normal"]))
    content.append(Paragraph(f"Total Records Analyzed: {len(df)}", styles["Normal"]))
    content.append(PageBreak())

    # ── 1. Executive Summary ─────────────────────────────────────────────────
    content.append(Paragraph("1. Executive Summary", heading1))
    content.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")))
    content.append(Spacer(1, 0.3 * cm))

    total = len(df)
    neg_count = (df.get("sentiment", pd.Series(["Neutral"] * total)) == "Negative").sum() if "sentiment" in df.columns else 0
    pos_count = (df.get("sentiment", pd.Series(["Neutral"] * total)) == "Positive").sum() if "sentiment" in df.columns else 0
    high_priority = (df.get("priority", pd.Series(["LOW"] * total)) == "HIGH").sum() if "priority" in df.columns else 0
    complaint_count = df.get("is_complaint", pd.Series([False] * total)).sum() if "is_complaint" in df.columns else 0

    summary_text = (
        f"This report presents the findings of the Customer Feedback Analysis Agent on a dataset of "
        f"<b>{total}</b> customer feedback records. The autonomous agent processed, analyzed, and "
        f"generated insights using a multi-stage agentic workflow. "
        f"Key findings include <b>{neg_count}</b> negative feedback records ({round(neg_count/total*100,1)}%), "
        f"<b>{high_priority}</b> high-priority issues requiring immediate action, "
        f"and <b>{len(recurring_issues)}</b> distinct recurring issue patterns identified through semantic clustering."
    )
    content.append(Paragraph(summary_text, body))
    content.append(Spacer(1, 0.3 * cm))

    # KPI Table
    kpi_data = [
        ["Metric", "Value"],
        ["Total Feedback Records", str(total)],
        ["Positive Feedback", f"{pos_count} ({round(pos_count/total*100,1) if total else 0}%)"],
        ["Negative Feedback", f"{neg_count} ({round(neg_count/total*100,1) if total else 0}%)"],
        ["High-Priority Issues", str(high_priority)],
        ["Total Complaints", str(complaint_count)],
        ["Recurring Issue Clusters", str(len(recurring_issues))],
    ]
    t = Table(kpi_data, colWidths=[10 * cm, 6 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f9f9f9"), colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    content.append(t)
    content.append(Spacer(1, 0.5 * cm))

    # ── 2. Dataset Overview ───────────────────────────────────────────────────
    content.append(Paragraph("2. Dataset Overview & Data Quality", heading1))
    content.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")))
    content.append(Spacer(1, 0.3 * cm))

    qa_data = [
        ["Data Quality Metric", "Count"],
        ["Records Received", str(quality_report.get("records_received", 0))],
        ["Duplicates Removed", str(quality_report.get("duplicates_removed", 0))],
        ["Missing Values Handled", str(quality_report.get("missing_handled", 0))],
        ["Short Records Removed", str(quality_report.get("short_removed", 0))],
        ["Valid Records Processed", str(quality_report.get("valid_records", 0))],
    ]
    t2 = Table(qa_data, colWidths=[10 * cm, 6 * cm])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f3460")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f0f4ff"), colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    content.append(t2)
    content.append(Spacer(1, 0.5 * cm))

    # ── 3. Sentiment Analysis ─────────────────────────────────────────────────
    content.append(Paragraph("3. Sentiment Analysis", heading1))
    content.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")))
    content.append(Spacer(1, 0.3 * cm))

    if "sentiment" in df.columns:
        sent_counts = df["sentiment"].value_counts()
        sent_data = [["Sentiment", "Count", "Percentage"]]
        for label in ["Positive", "Neutral", "Negative"]:
            count = sent_counts.get(label, 0)
            pct = round(count / total * 100, 1) if total else 0
            sent_data.append([label, str(count), f"{pct}%"])

        t3 = Table(sent_data, colWidths=[6 * cm, 5 * cm, 5 * cm])
        t3.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16213e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f9f9f9"), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        content.append(t3)
    else:
        content.append(Paragraph("Sentiment analysis data not available.", body))
    content.append(Spacer(1, 0.5 * cm))

    # ── 4. Recurring Issues ───────────────────────────────────────────────────
    content.append(Paragraph("4. Recurring Issues", heading1))
    content.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")))
    content.append(Spacer(1, 0.3 * cm))

    if recurring_issues:
        issue_data = [["Issue", "Occurrences", "%", "Sentiment", "Priority"]]
        for issue in recurring_issues[:10]:
            issue_data.append([
                issue["issue_label"],
                str(issue["occurrences"]),
                f"{issue['percentage']}%",
                issue["avg_sentiment"],
                issue["priority"],
            ])
        t4 = Table(issue_data, colWidths=[6 * cm, 3 * cm, 2 * cm, 3 * cm, 2 * cm])
        t4.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e94560")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#fff0f0"), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        content.append(t4)
    else:
        content.append(Paragraph("No significant recurring issues detected.", body))
    content.append(Spacer(1, 0.5 * cm))

    # ── 5. Key Insights ───────────────────────────────────────────────────────
    content.append(Paragraph("5. Key AI Insights", heading1))
    content.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")))
    content.append(Spacer(1, 0.3 * cm))

    for i, insight in enumerate(insights[:8], 1):
        content.append(Paragraph(f"{insight['icon']} {insight['title']}", heading2))
        content.append(Paragraph(insight["description"], body))
        content.append(Spacer(1, 0.2 * cm))

    content.append(Spacer(1, 0.3 * cm))

    # ── 6. Recommendations ────────────────────────────────────────────────────
    content.append(Paragraph("6. Recommended Actions", heading1))
    content.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")))
    content.append(Spacer(1, 0.3 * cm))

    for i, rec in enumerate(recommendations[:8], 1):
        priority_color = (
            colors.HexColor("#e94560") if rec["priority"] == "HIGH"
            else colors.HexColor("#f39c12") if rec["priority"] == "MEDIUM"
            else colors.HexColor("#27ae60")
        )
        content.append(Paragraph(
            f"<b>{i}. [{rec['priority']}] {rec['action']}</b>", heading2
        ))
        content.append(Paragraph(f"Issue: {rec['issue']} | Evidence: {rec['evidence']}", body))
        content.append(Paragraph(rec["details"], body))
        content.append(Paragraph(f"<b>KPI:</b> {rec['kpi']}", body))
        content.append(Paragraph(f"<b>Department:</b> {rec['department']}", body))
        content.append(Spacer(1, 0.3 * cm))

    # ── 7. Conclusion ─────────────────────────────────────────────────────────
    content.append(PageBreak())
    content.append(Paragraph("7. Conclusion", heading1))
    content.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")))
    content.append(Spacer(1, 0.3 * cm))

    conclusion = (
        "The Customer Feedback Analysis Agent successfully processed the submitted feedback dataset "
        "using an autonomous multi-tool agentic workflow. The agent identified key pain points, "
        "clustered recurring complaints, assigned data-driven priorities, and generated actionable "
        "recommendations. Implementation of the recommended actions, particularly for high-priority "
        "issues, is expected to significantly improve customer satisfaction and reduce complaint volume."
    )
    content.append(Paragraph(conclusion, body))
    content.append(Spacer(1, 1 * cm))
    content.append(Paragraph(
        "— Generated by Customer Feedback Analysis Agent —",
        ParagraphStyle("Footer", parent=styles["Normal"], alignment=TA_CENTER,
                       fontSize=9, textColor=colors.grey)
    ))

    doc.build(content)
    buffer.seek(0)
    return buffer.read()


def generate_pdf(
    df: pd.DataFrame,
    quality_report: Dict,
    recurring_issues: List[Dict],
    insights: List[Dict],
    recommendations: List[Dict],
    trend_data: Dict,
) -> Optional[bytes]:
    """Generate PDF report. Returns bytes or None if ReportLab unavailable."""
    if not _REPORTLAB_AVAILABLE:
        return None
    try:
        return _build_pdf(df, quality_report, recurring_issues, insights, recommendations, trend_data)
    except Exception as e:
        print(f"PDF generation error: {e}")
        return None


def generate_csv(df: pd.DataFrame) -> str:
    """Return analyzed data as CSV string."""
    # Drop non-serializable columns
    export_cols = [c for c in df.columns if c != "priority_reasons"]
    return df[export_cols].to_csv(index=False)
