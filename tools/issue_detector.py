"""
Issue Detector Tool
===================
Identifies the specific complaint in a feedback and maps it to a structured
issue label (used by the priority + recommendation engines).

Order matters: labels are listed most-specific first and ties are resolved in
that order.

When a `sentiment` column is available (it is, in the agent pipeline) the
detector uses it so that a *positive* review that merely mentions a minor
hiccup ("smooth refund, just a small delay") is not counted as a complaint.
"""

import re
from typing import Dict, Optional, Tuple

import pandas as pd

ISSUE_PATTERNS = {
    "Payment Issue": [
        r"(payment (failed|error|issue|problem|declined)|charged (twice|again)|double.?charged|"
        r"money (was )?deducted|amount (was )?deducted|billed twice|charged for (a|an|the) .{0,30}(never|not)|"
        r"never (actually )?ordered|invoice.{0,30}(discrepancy|wrong|mismatch|error)|"
        r"billing (error|issue|problem|mistake)|payment gateway|payment confirmation.{0,20}(long|late|delay|never))",
    ],
    "Pricing / Overcharge": [
        r"(overcharg\w*|charged (slightly |a lot )?more|higher than (what was |the )?(advertised|quoted|shown|price)|"
        r"hidden (fee|charge|cost)s?|extra (charge|cost|fee)s?|unexpected charge|price.{0,20}(too high|jumped|increase)|"
        r"not worth (the|its) price|overpriced)",
    ],
    "Refund Not Processed": [
        r"(refund.{0,40}(not|haven.t|hasn.t|never|still|pending|waiting|denied|rejected|delay|took|longer|no update)|"
        r"(waiting|wait|waited) (for|on) (my |the )?refund|no refund|want (a )?refund|need (a )?refund|"
        r"return.{0,40}(marked complete|never showed|label.{0,20}(didn.t|not) work|hasn.t happened|not (picked|collected))|"
        r"pickup.{0,30}return|return.{0,20}pickup.{0,40}(still|hasn.t|not)|sent back.{0,40}(no update|no response|nothing))",
    ],
    "Wrong Product": [
        r"(wrong (item|product|order|size|colou?r)|sent (me )?(the )?wrong|received (the )?wrong|"
        r"different (product|item) (from|than)|not what i ordered)",
    ],
    "Damaged Product": [
        r"(damaged|dented|crushed|cracked|torn|shattered|arrived (broken|damaged|smashed)|"
        r"visible defects?|item unusable|box was crushed|scratched on arrival|scuff\w* on arrival)",
    ],
    "Product Not Working": [
        r"(not working|stopped (working|functioning|charging)|doesn.t (work|turn on|charge|function)|"
        r"malfunction\w*|won.t (turn on|charge|connect|pair)|dead on arrival|broke (the )?(first|within|after)|"
        r"broke (down|on me)|stopped .{0,20}after (less than )?a (week|day|month))",
    ],
    "Poor Product Quality": [
        r"(poor (build )?quality|bad quality|low quality|cheap(ly)? (made|feel|plastic|material)|feels? (a bit )?(cheap|flimsy)|"
        r"flimsy|quality control|showing wear|wear and tear|scratch(es)? easily|rough finish|finish.{0,20}rough|"
        r"different quality|not as (described|shown|advertised)|completely different|falls apart|"
        r"not worth (the|its) price|poor (finish|material|build))",
    ],
    "Poor Customer Support": [
        r"(no (response|reply|resolution|fix|update|one (helped|replied|responded))|didn.t (respond|reply)|did not (respond|reply)|"
        r"(rude|unhelpful|useless|incompetent|dismissive).{0,30}(agent|support|staff|representative)|"
        r"(agent|support|staff|representative|customer service).{0,30}(rude|unhelpful|useless|incompetent|dismissive)|"
        r"(couldn.t|could not|can.t|unable to|failed to|never) (actually )?resolve|never got resolved|not (been )?resolved|"
        r"unresolved|transferred (between|from)|explain my issue (multiple|again|several)|had to (repeat|explain)|"
        r"called support \d+|contact(ing|ed) support|support (has not|hasn.t|did not|didn.t|never)|ignored my|"
        r"customer service.{0,20}(bad|poor|terrible|worst))",
    ],
    "Slow Customer Support": [
        r"(took \d+ days? to (respond|reply|answer)|slow(er)? (response|reply|to respond)|"
        r"response (eventually|felt slower|took)|long wait.{0,20}(support|agent|chat)|on hold for|"
        r"got a response eventually|slower than it should)",
    ],
    "Account Issue": [
        r"(can.t (update|change|edit|log ?in|sign in|access|reset)|unable to (log ?in|sign in|access|update)|"
        r"account.{0,15}(locked|blocked|suspended|hacked|problem|issue)|locked out|password.{0,20}(reset|not work|issue)|"
        r"otp.{0,20}(not|never|didn.t)|login.{0,15}(issue|problem|fail|error))",
    ],
    "App / Website Issue": [
        r"((app|website|site|checkout|search|page|notifications?).{0,60}(crash\w*|slow|laggy|lag|error|timed out|freez\w*|bug\w*|"
        r"glitch\w*|inaccurate|inconsistent|not (load|work|respond)\w*|down|broken|unresponsive)|"
        r"(crash\w*|laggy|timed out|glitch\w*|freez\w*).{0,40}(app|website|site|checkout|order)|"
        r"can.t (complete|finish) (my )?(checkout|order|payment)|keeps showing (an )?error|"
        r"(search|website|app).{0,30}(doesn.t|does not|not) (always )?(return|show|work|load))",
    ],
    "Missing / Lost Package": [
        r"(never (got|received|showed|arrived|came)|(haven.t|hasn.t|have not|has not|not) (yet )?(received|arrived)|"
        r"still haven.t received|package.{0,20}(lost|missing)|parcel.{0,20}(lost|missing)|lost (in transit|package|parcel)|"
        r"missing (item|package|parcel|part|accessor\w*)|marked (it )?(as )?delivered|wrong address)",
    ],
    "Delivery Delay": [
        r"(deliver\w*.{0,60}(late|delay\w*|slow|took \d+|instead of|only once|far away|not (on time|updated))|"
        r"(late|delayed|delay)\b.{0,30}(deliver\w*|shipping|shipment|package|parcel|order|courier)|"
        r"(package|parcel|order|shipment).{0,40}(late|delay\w*|took \d+ days|showed up)|"
        r"(shipping|delivery|courier|tracking).{0,50}(wasn.t updated|not updated|no updates?|standard (delivery|shipping)|only once|left it)|"
        r"still waiting for (my )?(order|package|delivery|parcel)|took (too long|longer than promised)|arrived (very )?late|"
        r"paid extra for express)",
    ],
    "Order Cancellation": [
        r"(order.{0,20}cancel\w*|cancellation|cancelled (it )?without|automatically cancel\w*)",
    ],
    "General Complaint": [
        r"(bad experience|terrible experience|horrible experience|frustrating experience|worst.{0,20}(ever|experience)|"
        r"very disappointed|extremely dissatisfied|underwhelming|letdown|let down|forgettable|more hassle than|"
        r"(would not|wouldn.t|won.t|will not|not planning to) (recommend|order|purchase|buy)|never (buy|order|shop)\w* (from|again))",
    ],
}

_COMPILED = {issue: [re.compile(p) for p in pats] for issue, pats in ISSUE_PATTERNS.items()}


def _norm(text: str) -> str:
    return str(text).lower().replace("\u2019", "'").replace("\u2018", "'")


def detect_issue(text: str, sentiment: Optional[str] = None) -> Tuple[str, float]:
    """Detect the primary issue in a feedback text."""
    t = _norm(text)
    matched: Dict[str, int] = {}
    for issue, pats in _COMPILED.items():
        for pat in pats:
            if pat.search(t):
                matched[issue] = matched.get(issue, 0) + 1

    if sentiment == "Positive":
        # A happy customer isn't reporting an issue
        return "General Feedback", 45.0

    if not matched:
        if sentiment == "Negative":
            return "General Complaint", 55.0
        return "General Feedback", 45.0

    best = max(matched, key=matched.get)   # ties -> first (most specific) label
    confidence = min(92.0, 60 + matched[best] * 12)
    if best == "General Complaint" and len(matched) > 1:
        # a specific issue exists as well -> prefer it
        matched.pop(best)
        best = max(matched, key=matched.get)
        confidence = min(92.0, 60 + matched[best] * 12)
    return best, round(confidence, 1)


def is_complaint(text: str, sentiment: Optional[str] = None) -> bool:
    """Binary check whether the feedback is a complaint."""
    if sentiment == "Negative":
        return True
    if sentiment == "Positive":
        return False
    issue, confidence = detect_issue(text, sentiment)
    return issue != "General Feedback" and confidence >= 55


def run(df: pd.DataFrame, feedback_col: str) -> pd.DataFrame:
    """Run issue detection on all feedback rows."""
    df = df.copy()
    sentiments = df["sentiment"].tolist() if "sentiment" in df.columns else [None] * len(df)
    issues, confs, flags = [], [], []
    for text, sent in zip(df[feedback_col].astype(str), sentiments):
        issue, conf = detect_issue(text, sent)
        issues.append(issue)
        confs.append(conf)
        flags.append(is_complaint(text, sent))
    df["detected_issue"] = issues
    df["issue_confidence"] = confs
    df["is_complaint"] = flags
    return df


def get_issue_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize detected issues with counts, average sentiment and priority."""
    if "detected_issue" not in df.columns:
        return pd.DataFrame()

    summary = (
        df.groupby("detected_issue")
        .agg(occurrences=("detected_issue", "count"),
             avg_confidence=("issue_confidence", "mean"))
        .reset_index()
    )
    summary = summary[summary["detected_issue"] != "General Feedback"]
    summary["percentage"] = (summary["occurrences"] / max(len(df), 1) * 100).round(1)

    if "sentiment" in df.columns and len(summary):
        sentiment_map = df.groupby("detected_issue")["sentiment"].agg(
            lambda x: x.value_counts().index[0]
        )
        summary["avg_sentiment"] = summary["detected_issue"].map(sentiment_map)

    return summary.sort_values("occurrences", ascending=False).reset_index(drop=True)
