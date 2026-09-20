"""
Recommendation Engine
=====================
Converts detected issues and insights into actionable business recommendations.
All recommendations are evidence-based from actual analysis data.
"""

from typing import List, Dict
import pandas as pd


ISSUE_RECOMMENDATIONS = {
    "Delivery Delay": {
        "action": "Review Logistics Partner Performance",
        "details": (
            "Audit delivery SLAs with current logistics partners. "
            "Implement real-time shipment tracking and proactive delay notifications. "
            "Consider diversifying to multiple delivery partners to reduce bottlenecks."
        ),
        "kpi": "Target: Reduce average delivery time by 30% within 60 days.",
        "department": "Supply Chain / Operations",
    },
    "Missing / Lost Package": {
        "action": "Strengthen Package Tracking & Insurance",
        "details": (
            "Mandate GPS tracking for all shipments above a threshold value. "
            "Introduce automatic refund/reship policy for packages lost in transit. "
            "Audit high-loss delivery routes and carriers."
        ),
        "kpi": "Target: Reduce lost package complaints by 80% within 90 days.",
        "department": "Logistics / Customer Success",
    },
    "Damaged Product": {
        "action": "Improve Packaging Standards",
        "details": (
            "Conduct packaging stress tests for fragile product categories. "
            "Train warehouse staff on proper packing procedures. "
            "Introduce double-boxing for high-value or fragile items."
        ),
        "kpi": "Target: Reduce product damage complaints by 60% within 45 days.",
        "department": "Warehouse / Quality Assurance",
    },
    "Wrong Product": {
        "action": "Enhance Order Verification Process",
        "details": (
            "Implement barcode scanning at packing stations to verify every item. "
            "Introduce quality-check step before dispatch for large orders. "
            "Audit order processing system for known error patterns."
        ),
        "kpi": "Target: Achieve 99.5% order accuracy within 30 days.",
        "department": "Warehouse Operations",
    },
    "Refund Not Processed": {
        "action": "Streamline Refund Processing Pipeline",
        "details": (
            "Automate refund initiation upon return confirmation. "
            "Set a maximum 3-business-day SLA for refund processing. "
            "Create a dedicated refund status tracker visible to customers."
        ),
        "kpi": "Target: Process 95% of refunds within 3 business days.",
        "department": "Finance / Customer Support",
    },
    "Payment Issue": {
        "action": "Audit Payment Gateway & Billing System",
        "details": (
            "Conduct a technical audit of the payment gateway for double-charge scenarios. "
            "Implement automatic reconciliation to detect billing anomalies. "
            "Provide instant refund for verified overcharges within 24 hours."
        ),
        "kpi": "Target: Resolve all payment anomalies within 24 hours.",
        "department": "Finance / Engineering",
    },
    "Poor Customer Support": {
        "action": "Improve Customer Support Quality & Training",
        "details": (
            "Conduct mandatory soft-skills training for all support agents. "
            "Implement customer satisfaction (CSAT) scoring after each interaction. "
            "Establish a quality assurance review of 10% of all support tickets weekly."
        ),
        "kpi": "Target: Achieve CSAT score ≥ 4.0/5.0 within 60 days.",
        "department": "Customer Support",
    },
    "Slow Customer Support": {
        "action": "Reduce Support Response Time",
        "details": (
            "Review and enforce first-response SLA (target: 2 hours for email, 5 min for chat). "
            "Increase support staff during identified peak hours. "
            "Implement AI-based ticket routing for faster resolution."
        ),
        "kpi": "Target: First response time under 2 hours for 90% of tickets.",
        "department": "Customer Support Operations",
    },
    "Product Not Working": {
        "action": "Investigate & Fix Product Reliability Issues",
        "details": (
            "Conduct a product quality audit for reported SKUs. "
            "Introduce mandatory QA testing before batch shipment. "
            "Create a fast-track replacement program for defective products."
        ),
        "kpi": "Target: Reduce product defect complaints by 70% within 90 days.",
        "department": "Product Quality / Engineering",
    },
    "Poor Product Quality": {
        "action": "Enhance Quality Control Standards",
        "details": (
            "Tighten supplier quality standards and conduct random batch testing. "
            "Introduce customer-reported quality feedback loop into production cycle. "
            "Consider third-party quality certification for key product lines."
        ),
        "kpi": "Target: Reduce quality complaints by 50% within 90 days.",
        "department": "Procurement / Quality Assurance",
    },
    "App / Website Issue": {
        "action": "Prioritize Platform Stability & UX Improvements",
        "details": (
            "Conduct a comprehensive performance audit of the app and website. "
            "Fix identified crash points and implement error monitoring (e.g., Sentry). "
            "Run monthly UX testing sessions with real users."
        ),
        "kpi": "Target: Achieve 99.9% uptime and reduce crash reports by 80%.",
        "department": "Engineering / Product",
    },
    "Account Issue": {
        "action": "Simplify Account & Authentication Flow",
        "details": (
            "Streamline the password reset and OTP verification process. "
            "Implement social login (Google/Apple) to reduce friction. "
            "Add 24/7 account recovery support for locked accounts."
        ),
        "kpi": "Target: Reduce account-related support tickets by 60%.",
        "department": "Engineering / Customer Support",
    },
    "Pricing / Overcharge": {
        "action": "Review & Communicate Pricing Policy Clearly",
        "details": (
            "Audit all pricing tiers and hidden fees. "
            "Display total cost breakdown clearly during checkout. "
            "Send proactive communication about any pricing changes."
        ),
        "kpi": "Target: Reduce pricing complaint rate to below 2%.",
        "department": "Finance / Marketing",
    },
    "Order Cancellation": {
        "action": "Improve Order Stability & Communication",
        "details": (
            "Review triggers for automatic order cancellation and fix false positives. "
            "Notify customers immediately with clear reasons for any cancellation. "
            "Provide an easy order reinstatement flow."
        ),
        "kpi": "Target: Reduce unintended cancellations by 90%.",
        "department": "Operations / Engineering",
    },
    "General Complaint": {
        "action": "Establish Structured Customer Feedback Response Process",
        "details": (
            "Implement a systematic process for reviewing and responding to all complaints. "
            "Create a customer experience task force to address recurring pain points. "
            "Track and report complaint resolution rate monthly."
        ),
        "kpi": "Target: Respond to 100% of complaints within 48 hours.",
        "department": "Customer Experience",
    },
}

DEFAULT_RECOMMENDATION = {
    "action": "Monitor and Address Emerging Issue",
    "details": "Conduct further analysis of this issue cluster and assign it to the relevant team.",
    "kpi": "Track resolution rate monthly.",
    "department": "Customer Experience",
}


def generate_recommendations(
    recurring_issues: List[Dict],
    issue_summary: pd.DataFrame = None,
) -> List[Dict]:
    """
    Generate actionable recommendations from recurring issues.

    Returns list of recommendation dicts sorted by priority.
    """
    recommendations = []
    seen_actions = set()

    # From recurring issues
    for issue in recurring_issues:
        label = issue["issue_label"]
        rec_template = ISSUE_RECOMMENDATIONS.get(label, DEFAULT_RECOMMENDATION)

        if rec_template["action"] in seen_actions:
            continue
        seen_actions.add(rec_template["action"])

        recommendations.append({
            "issue": label,
            "occurrences": issue["occurrences"],
            "percentage": issue["percentage"],
            "priority": issue["priority"],
            "avg_sentiment": issue.get("avg_sentiment", "Negative"),
            "action": rec_template["action"],
            "details": rec_template["details"],
            "kpi": rec_template["kpi"],
            "department": rec_template["department"],
            "evidence": (
                f"{issue['occurrences']} occurrences detected "
                f"({issue['percentage']}% of all feedback) with "
                f"{issue.get('avg_sentiment', 'Negative')} average sentiment."
            ),
        })

    # From issue summary if available
    if issue_summary is not None and len(issue_summary) > 0:
        for _, row in issue_summary.iterrows():
            label = row.get("detected_issue", "")
            if not label or label == "General Feedback":
                continue
            rec_template = ISSUE_RECOMMENDATIONS.get(label, DEFAULT_RECOMMENDATION)
            if rec_template["action"] in seen_actions:
                continue
            if row.get("occurrences", 0) < 3:
                continue
            seen_actions.add(rec_template["action"])
            recommendations.append({
                "issue": label,
                "occurrences": int(row.get("occurrences", 0)),
                "percentage": float(row.get("percentage", 0)),
                "priority": row.get("priority", "MEDIUM"),
                "avg_sentiment": row.get("avg_sentiment", "Neutral"),
                "action": rec_template["action"],
                "details": rec_template["details"],
                "kpi": rec_template["kpi"],
                "department": rec_template["department"],
                "evidence": (
                    f"{int(row.get('occurrences', 0))} occurrences detected "
                    f"({float(row.get('percentage', 0))}% of feedback)."
                ),
            })

    # Sort by priority
    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    recommendations.sort(key=lambda x: priority_order.get(x["priority"], 3))
    return recommendations
