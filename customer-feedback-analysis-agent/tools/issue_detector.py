"""
Issue Detector Tool
===================
Identifies the specific complaint/issue in a feedback.
Maps detected patterns to structured issue labels.
"""

import re
import pandas as pd
from typing import Dict, List, Tuple

ISSUE_PATTERNS = {
    "Delivery Delay": [
        r"(late|delayed|delay|didn.t arrive|not arrived|still waiting|waiting for|took too long|took \d+ days|hasn.t come|never arrived|never came|not yet arrived)",
        r"(delivery.*late|shipping.*delay|order.*not.*received|package.*delay|arrival.*delay)"
    ],
    "Missing / Lost Package": [
        r"(missing|lost|never received|not received|didn.t receive|package.*lost|parcel.*missing|never arrived|never showed up|still not received)"
    ],
    "Damaged Product": [
        r"(damaged|broken|defective|faulty|cracked|torn|scratched|dented|arrived.*broken|item.*damaged|product.*damage)"
    ],
    "Wrong Product": [
        r"(wrong (item|product|order)|sent wrong|received wrong|different (product|item)|ordered.*got.*different)"
    ],
    "Refund Not Processed": [
        r"(refund.*(not|haven.t|still|pending|waiting)|waiting for refund|no refund|refund denied|refund rejected|want refund|need refund)"
    ],
    "Payment Issue": [
        r"(payment failed|payment.*(error|issue|problem)|charged twice|double charged|money deducted.*not|amount deducted|overcharged|billed twice)"
    ],
    "Poor Customer Support": [
        r"(no response|no reply|didn.t respond|did not respond|support.*(rude|unhelpful|useless|bad|terrible|worst)|agent.*(rude|useless|unhelpful)|not helpful|customer service.*(bad|terrible|poor|worst|rude))",
        r"(never got.*response|ignored my|dismissed my|complaint.*ignored|customer service.*not respond)"
    ],
    "Slow Customer Support": [
        r"(waiting for.*hours|waiting for.*days|long wait|took.*days to respond|slow response|response.*late|no reply in|didn.t respond|took.*long to.*reply)"
    ],
    "Product Not Working": [
        r"(not working|stopped working|doesn.t work|malfunctioning|broken.*product|product.*broken|won.t turn on|doesn.t function|stopped functioning)"
    ],
    "Poor Product Quality": [
        r"(poor quality|bad quality|cheap quality|low quality|quality.*bad|not good quality|terrible quality|worst quality)"
    ],
    "App / Website Issue": [
        r"(app.*(crash|not working|error|bug|slow|freeze)|website.*(down|not loading|slow|error|crash)|page.*not.*load|can.t.*checkout|checkout.*fail)"
    ],
    "Account Issue": [
        r"(can.t login|login.*issue|account.*locked|password.*reset|can.t access account|sign in.*fail|unable to login|account.*problem)"
    ],
    "Pricing / Overcharge": [
        r"(overcharged|charged more|price.*high|expensive.*than|hidden (fee|charge)|extra (charge|cost)|unexpected charge)"
    ],
    "Order Cancellation": [
        r"(order.*cancel|cancellation|cancelled.*without|automatically cancel|order.*cancelled)"
    ],
    "General Complaint": [
        r"(bad experience|terrible experience|worst.*ever|very disappointed|extremely dissatisfied|will not.*again|never.*again|horrible experience)"
    ],
}



def detect_issue(text: str) -> Tuple[str, float]:
    """Detect the primary issue in feedback text."""
    text_lower = text.lower()
    matched_issues: Dict[str, int] = {}

    for issue, patterns in ISSUE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                matched_issues[issue] = matched_issues.get(issue, 0) + 1

    if not matched_issues:
        # Light heuristic: check topic to guess issue
        return "General Feedback", 45.0

    best_issue = max(matched_issues, key=matched_issues.get)
    confidence = min(92.0, 60 + matched_issues[best_issue] * 12)
    return best_issue, round(confidence, 1)


def is_complaint(text: str) -> bool:
    """Quick binary check if feedback is a complaint."""
    issue, confidence = detect_issue(text)
    return issue != "General Feedback" and confidence >= 55


def run(df: pd.DataFrame, feedback_col: str) -> pd.DataFrame:
    """Run issue detection on all feedback rows."""
    results = df[feedback_col].apply(
        lambda t: pd.Series({
            "detected_issue": detect_issue(str(t))[0],
            "issue_confidence": detect_issue(str(t))[1],
            "is_complaint": is_complaint(str(t))
        })
    )
    df = df.copy()
    for col in results.columns:
        df[col] = results[col].values
    return df


def get_issue_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize detected issues with counts and average sentiment."""
    if "detected_issue" not in df.columns:
        return pd.DataFrame()

    summary = (
        df.groupby("detected_issue")
        .agg(
            occurrences=("detected_issue", "count"),
            avg_confidence=("issue_confidence", "mean"),
        )
        .reset_index()
    )
    summary = summary[summary["detected_issue"] != "General Feedback"]
    summary["percentage"] = (summary["occurrences"] / len(df) * 100).round(1)

    if "sentiment" in df.columns:
        sentiment_map = df.groupby("detected_issue")["sentiment"].agg(
            lambda x: x.value_counts().index[0]
        )
        summary["avg_sentiment"] = summary["detected_issue"].map(sentiment_map)

    summary = summary.sort_values("occurrences", ascending=False).reset_index(drop=True)
    return summary
