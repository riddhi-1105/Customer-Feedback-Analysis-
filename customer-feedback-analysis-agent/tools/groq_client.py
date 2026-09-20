"""
Groq LLM Integration Module
===========================
Provides high-speed LLM enhancement using Groq's API for the
Customer Feedback Analysis Agent.

If GROQ_API_KEY is not configured or fails, all callers automatically
fall back to the local statistical/rule-based NLP engines.
"""

import os
import json
import logging
from typing import Optional, List, Dict, Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from groq import Groq
    GROQ_INSTALLED = True
except ImportError:
    GROQ_INSTALLED = False

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def get_groq_api_key() -> Optional[str]:
    """Retrieve the Groq API key from environment variables."""
    key = os.getenv("GROQ_API_KEY", "").strip()
    return key if key else None


def is_groq_available(api_key: Optional[str] = None) -> bool:
    """Check if Groq SDK is installed and a valid API key is present."""
    if not GROQ_INSTALLED:
        return False
    key = api_key or get_groq_api_key()
    return bool(key and len(key) > 5)


def get_groq_client(api_key: Optional[str] = None) -> Optional[Any]:
    """Instantiate a Groq client if key is available."""
    if not GROQ_INSTALLED:
        return None
    key = api_key or get_groq_api_key()
    if not key:
        return None
    try:
        return Groq(api_key=key)
    except Exception as e:
        logger.warning(f"Failed to initialize Groq client: {e}")
        return None


def explain_feedback_with_groq(
    feedback_text: str,
    sentiment: str,
    topic: str,
    issue: str,
    priority: str,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL
) -> Optional[str]:
    """
    Use Groq LLM to generate an executive root-cause breakdown and
    resolution plan for a single feedback record.
    """
    client = get_groq_client(api_key)
    if not client:
        return None

    prompt = f"""You are a Principal Customer Experience AI Agent. Analyze the following customer feedback:
Feedback: "{feedback_text}"
Detected Sentiment: {sentiment}
Detected Topic: {topic}
Detected Issue: {issue}
Urgency Priority: {priority}

Provide a concise, highly practical executive analysis in exactly 3-4 bullet points:
1. Underlying Root Cause: Why did this problem occur?
2. Customer Emotional Impact: How does this affect churn/loyalty?
3. Recommended Immediate Action: What should customer support or engineering do right now?
4. Preventative Measure: Long-term fix to ensure this does not repeat.

Keep the response professional, concise, and direct."""

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an elite customer intelligence and root-cause analysis agent."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=400,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Groq explain_feedback call failed: {e}")
        return None


def generate_llm_executive_summary(
    metrics: Dict[str, Any],
    top_issues: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL
) -> Optional[Dict[str, Any]]:
    """
    Use Groq LLM to synthesize high-level strategic intelligence from
    dataset metrics and top issues.
    """
    client = get_groq_client(api_key)
    if not client:
        return None

    prompt = f"""You are a Chief Customer Officer AI Agent. Given the feedback analysis summary:
- Total Analyzed Records: {metrics.get('total', 0)}
- Sentiment Split: Positive {metrics.get('positive_pct', 0)}%, Negative {metrics.get('negative_pct', 0)}%, Neutral {metrics.get('neutral_pct', 0)}%
- Top Recurring Issues: {json.dumps(top_issues[:5])}

Generate an executive debrief in valid JSON format with this exact schema:
{{
  "executive_headline": "Short punchy 1-sentence headline",
  "strategic_diagnosis": "2-3 sentences diagnosing the core operational friction points",
  "key_priorities": [
    "Priority action 1",
    "Priority action 2",
    "Priority action 3"
  ]
}}
Output valid JSON only. No markdown formatting, no commentary."""

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You output valid raw JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=450,
            response_format={"type": "json_object"}
        )
        content = completion.choices[0].message.content.strip()
        return json.loads(content)
    except Exception as e:
        logger.warning(f"Groq executive summary call failed: {e}")
        return None
