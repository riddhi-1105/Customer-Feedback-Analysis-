"""
Groq LLM integration (optional). Every caller falls back to the local engines
if no key is set or a call fails.

Key lookup order: explicit argument (per-user session key) ->
Streamlit secrets -> environment variable / .env.
Model fallback chain protects against a retired/misspelled model name.
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

DEFAULT_MODEL = os.getenv("GROQ_MODEL", "").strip() or "llama-3.3-70b-versatile"
FALLBACK_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "openai/gpt-oss-120b"]


def get_groq_api_key() -> Optional[str]:
    key = ""
    try:
        import streamlit as st
        key = str(st.secrets.get("GROQ_API_KEY", "")).strip()
    except Exception:
        pass
    key = key or os.getenv("GROQ_API_KEY", "").strip()
    return key or None


def is_groq_available(api_key: Optional[str] = None) -> bool:
    if not GROQ_INSTALLED:
        return False
    key = api_key or get_groq_api_key()
    return bool(key and len(key) > 5)


def get_groq_client(api_key: Optional[str] = None) -> Optional[Any]:
    if not GROQ_INSTALLED:
        return None
    key = api_key or get_groq_api_key()
    if not key:
        return None
    try:
        return Groq(api_key=key)
    except Exception as e:
        logger.warning("Failed to initialize Groq client: %s", e)
        return None


def _chat(client, model: str, **kwargs):
    """Try the requested model, then the fallback chain."""
    last = None
    for m in [model] + [x for x in FALLBACK_MODELS if x != model]:
        try:
            return client.chat.completions.create(model=m, **kwargs)
        except Exception as e:
            last = e
            logger.warning("Groq model %s failed: %s", m, e)
    raise last


def explain_feedback_with_groq(feedback_text: str, sentiment: str, topic: str, issue: str,
                               priority: str, api_key: Optional[str] = None,
                               model: str = DEFAULT_MODEL) -> Optional[str]:
    client = get_groq_client(api_key)
    if not client:
        return None
    prompt = f"""Analyze this customer feedback:
Feedback: "{feedback_text}"
Sentiment: {sentiment} | Topic: {topic} | Issue: {issue} | Priority: {priority}

Reply in exactly 4 short bullets:
1. Underlying Root Cause
2. Customer Emotional Impact (churn/loyalty)
3. Recommended Immediate Action
4. Preventative Measure"""
    try:
        c = _chat(client, model, messages=[
            {"role": "system", "content": "You are a customer-experience root-cause analyst."},
            {"role": "user", "content": prompt}], temperature=0.2, max_tokens=400)
        return c.choices[0].message.content.strip()
    except Exception as e:
        logger.warning("Groq explain failed: %s", e)
        return None


def generate_llm_executive_summary(metrics: Dict[str, Any], top_issues: List[Dict[str, Any]],
                                   api_key: Optional[str] = None,
                                   model: str = DEFAULT_MODEL) -> Optional[Dict[str, Any]]:
    client = get_groq_client(api_key)
    if not client:
        return None
    slim = [{"issue": i.get("issue_label"), "count": i.get("occurrences")} for i in top_issues[:5]]
    prompt = f"""Feedback analysis summary:
- Records: {metrics.get('total', 0)}
- Positive {metrics.get('positive_pct', 0)}%, Negative {metrics.get('negative_pct', 0)}%, Neutral {metrics.get('neutral_pct', 0)}%
- Top recurring issues: {json.dumps(slim)}

Return JSON only: {{"executive_headline": "...", "strategic_diagnosis": "2-3 sentences", "key_priorities": ["...","...","..."]}}"""
    try:
        c = _chat(client, model, messages=[
            {"role": "system", "content": "You output valid raw JSON only."},
            {"role": "user", "content": prompt}], temperature=0.2, max_tokens=450,
            response_format={"type": "json_object"})
        return json.loads(c.choices[0].message.content.strip())
    except Exception as e:
        logger.warning("Groq summary failed: %s", e)
        return None
