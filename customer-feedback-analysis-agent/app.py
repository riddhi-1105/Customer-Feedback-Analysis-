"""
Customer Feedback Analysis Agent
=================================
Main Streamlit Application

Run with: streamlit run app.py
"""

import os
import sys
import time
import json
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from agents.customer_feedback_agent import CustomerFeedbackAgent
from utils.helpers import (
    detect_feedback_column, detect_date_column, detect_rating_column,
    sentiment_pie_chart, emotion_bar_chart, topic_bar_chart,
    priority_donut_chart, sentiment_trend_chart, volume_trend_chart,
    recurring_issues_chart,
    COLORS, SENTIMENT_COLORS, EMOTION_COLORS, PRIORITY_COLORS
)
from tools import report_generator
from tools.groq_client import (
    is_groq_available, explain_feedback_with_groq,
    generate_llm_executive_summary, DEFAULT_MODEL
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Feedback Analysis Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS (Light Wine & Cream Palette) ──────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* Color Variables - Light Wine & Cream */
:root {
    --bg-cream: #FAF6F0;
    --bg-surface: rgba(255, 255, 255, 0.9);
    --bg-surface-elevated: #FFFFFF;
    --wine-primary: #722F37;      /* Classic Wine / Bordeaux */
    --wine-deep: #4A1521;         /* Deep Velvet Wine */
    --wine-light: #A3485E;        /* Soft Rosé Wine */
    --wine-pale: #F3EAEB;         /* Delicate Wine Wash */
    --gold-accent: #C5A059;       /* Champagne Gold */
    --sage-green: #2D7A58;        /* Botanical Emerald */
    --amber-warm: #BA6A24;        /* Warm Terracotta Sienna */
    --crimson-wine: #9C1D3A;      /* Alert Wine Crimson */
    --text-primary: #33181E;      /* Deep Wine Espresso */
    --text-secondary: #6E4D53;    /* Muted Wine Mauve */
    --text-muted: #8C6C72;        /* Soft Taupe Rose */
    --border-wine: rgba(114, 47, 55, 0.12);
    --border-wine-strong: rgba(114, 47, 55, 0.24);
}

html, body, [class*="css"], .stMarkdown {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text-primary) !important;
}

/* Warm Cream Background with Delicate Ambient Rosé Tint */
.stApp {
    background: 
        radial-gradient(circle at 50% -10%, #F5EDE6 0%, #FAF6F0 45%, #F4ECE3 100%) !important;
    background-attachment: fixed !important;
    color: var(--text-primary) !important;
}

/* Container Spacing & Layout */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 4rem !important;
    max-width: 1440px !important;
}

/* Custom Scrollbars */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #FAF6F0;
}
::-webkit-scrollbar-thumb {
    background: rgba(114, 47, 55, 0.25);
    border-radius: 9999px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(114, 47, 55, 0.55);
}

/* Hide Streamlit Boilerplate */
#MainMenu, footer, header {
    visibility: hidden !important;
}

/* Sidebar Styling - Porcelain Cream with Wine Border */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FDFBF8 0%, #F6EEE6 100%) !important;
    border-right: 1px solid rgba(114, 47, 55, 0.14) !important;
    box-shadow: 6px 0 24px rgba(114, 47, 55, 0.04) !important;
}

[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

/* Sidebar Brand Header */
.brand-container {
    padding: 8px 0 20px 0;
    text-align: left;
    border-bottom: 1px solid rgba(114, 47, 55, 0.1);
    margin-bottom: 16px;
}
.brand-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: linear-gradient(135deg, rgba(114, 47, 55, 0.08), rgba(163, 72, 94, 0.12));
    border: 1px solid rgba(114, 47, 55, 0.22);
    padding: 5px 12px;
    border-radius: 9999px;
    font-size: 11px;
    font-weight: 700;
    color: #722F37 !important;
    letter-spacing: 0.6px;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.brand-logo {
    font-family: 'Outfit', sans-serif !important;
    font-size: 24px;
    font-weight: 800;
    background: linear-gradient(135deg, #4A1521 0%, #722F37 50%, #A3485E 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
    line-height: 1.2;
}
.brand-sub {
    font-size: 12px;
    color: #8C6C72 !important;
    font-weight: 500;
    margin-top: 4px;
}

/* Sidebar Radio Navigation - Elegant Cream Pills */
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
    gap: 6px;
    display: flex;
    flex-direction: column;
}

[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label {
    background: rgba(255, 255, 255, 0.75);
    border: 1px solid rgba(114, 47, 55, 0.09);
    border-radius: 10px;
    padding: 10px 14px;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    margin-bottom: 2px;
}

[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover {
    background: rgba(114, 47, 55, 0.08);
    border-color: rgba(114, 47, 55, 0.25);
    transform: translateX(3px);
}

[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover span {
    color: #4A1521 !important;
    font-weight: 600;
}

/* Status Pill in Sidebar */
.status-pill {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 14px;
    border-radius: 12px;
    margin-top: 12px;
    transition: all 0.25s ease;
}
.status-pill.ready {
    background: rgba(114, 47, 55, 0.06);
    border: 1px solid rgba(114, 47, 55, 0.2);
}
.status-pill.complete {
    background: rgba(45, 122, 88, 0.08);
    border: 1px solid rgba(45, 122, 88, 0.25);
}
.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    animation: pulse-ring 2s cubic-bezier(0.455, 0.03, 0.515, 0.955) infinite;
}
.status-dot.ready { background: #722F37; box-shadow: 0 0 10px rgba(114, 47, 55, 0.5); }
.status-dot.complete { background: #2D7A58; box-shadow: 0 0 10px rgba(45, 122, 88, 0.5); }

@keyframes pulse-ring {
    0% { transform: scale(0.9); opacity: 0.8; }
    50% { transform: scale(1.3); opacity: 1; }
    100% { transform: scale(0.9); opacity: 0.8; }
}

/* Modern Hero Section - Cream & Wine Silk */
.hero-wrapper {
    position: relative;
    background: linear-gradient(145deg, rgba(255, 255, 255, 0.94) 0%, rgba(250, 244, 238, 0.92) 100%);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(114, 47, 55, 0.16);
    border-radius: 20px;
    padding: 44px 36px;
    margin-bottom: 28px;
    box-shadow: 0 20px 45px -15px rgba(114, 47, 55, 0.1), 0 4px 12px rgba(0, 0, 0, 0.02);
    overflow: hidden;
    text-align: center;
}

.hero-wrapper::before {
    content: '';
    position: absolute;
    top: -100px;
    left: 50%;
    transform: translateX(-50%);
    width: 380px;
    height: 220px;
    background: radial-gradient(circle, rgba(163, 72, 94, 0.12) 0%, transparent 70%);
    filter: blur(50px);
    pointer-events: none;
}

.hero-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(114, 47, 55, 0.08);
    border: 1px solid rgba(114, 47, 55, 0.22);
    padding: 6px 16px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 700;
    color: #722F37;
    letter-spacing: 0.5px;
    margin-bottom: 18px;
    box-shadow: 0 2px 8px rgba(114, 47, 55, 0.06);
}

.hero-title {
    font-family: 'Outfit', sans-serif !important;
    font-size: 46px;
    font-weight: 900;
    letter-spacing: -1.2px;
    line-height: 1.15;
    margin-bottom: 14px;
    color: #380E16;
}

.gradient-text {
    background: linear-gradient(135deg, #380E16 0%, #722F37 50%, #A3485E 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-desc {
    font-size: 16px;
    color: #6E4D53;
    max-width: 680px;
    margin: 0 auto 28px auto;
    line-height: 1.6;
}

.hero-stats-row {
    display: flex;
    justify-content: center;
    gap: 16px;
    flex-wrap: wrap;
    margin-top: 10px;
}

.hero-stat-badge {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(255, 255, 255, 0.85);
    border: 1px solid rgba(114, 47, 55, 0.12);
    border-radius: 12px;
    padding: 8px 16px;
    font-size: 13px;
    color: #4A1521;
    font-weight: 600;
    box-shadow: 0 2px 6px rgba(114, 47, 55, 0.04);
}

/* Cream & Wine Frosted KPI Cards */
.kpi-card {
    position: relative;
    background: linear-gradient(145deg, rgba(255, 255, 255, 0.95) 0%, rgba(252, 248, 244, 0.9) 100%);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(114, 47, 55, 0.12);
    border-radius: 16px;
    padding: 22px 18px;
    text-align: center;
    box-shadow: 0 10px 25px -8px rgba(114, 47, 55, 0.08), 0 2px 6px rgba(0, 0, 0, 0.02);
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    overflow: hidden;
    margin-bottom: 12px;
}

.kpi-card:hover {
    transform: translateY(-4px);
    border-color: rgba(114, 47, 55, 0.35);
    box-shadow: 0 16px 35px -10px rgba(114, 47, 55, 0.18), 0 4px 10px rgba(0, 0, 0, 0.03);
}

.kpi-title {
    font-size: 12px;
    color: #8C6C72;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
}

.kpi-value {
    font-family: 'Outfit', sans-serif;
    font-size: 34px;
    font-weight: 800;
    color: #380E16;
    line-height: 1.1;
    letter-spacing: -0.5px;
}

.kpi-sub {
    font-size: 12px;
    color: #6E4D53;
    margin-top: 6px;
    font-weight: 500;
}

/* Section Header with Wine Underline */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    font-family: 'Outfit', sans-serif !important;
    font-size: 23px;
    font-weight: 700;
    color: #380E16;
    letter-spacing: -0.4px;
    margin: 28px 0 16px 0;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(114, 47, 55, 0.14);
}

/* Feature Cards with Soft Rose Glow */
.feature-card {
    background: linear-gradient(145deg, rgba(255, 255, 255, 0.95) 0%, rgba(252, 248, 244, 0.9) 100%);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(114, 47, 55, 0.12);
    border-radius: 16px;
    padding: 24px;
    margin: 8px 0;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    box-shadow: 0 8px 20px rgba(114, 47, 55, 0.06);
    height: calc(100% - 16px);
}

.feature-card:hover {
    border-color: rgba(114, 47, 55, 0.38);
    transform: translateY(-4px);
    box-shadow: 0 16px 32px -8px rgba(114, 47, 55, 0.16);
}

.feature-icon {
    font-size: 32px;
    margin-bottom: 14px;
    display: inline-block;
}

.feature-title {
    font-family: 'Outfit', sans-serif;
    font-size: 17px;
    font-weight: 700;
    color: #380E16;
    margin-bottom: 8px;
    letter-spacing: -0.2px;
}

.feature-desc {
    font-size: 13.5px;
    color: #6E4D53;
    line-height: 1.55;
}

/* Priority Badges */
.badge-HIGH {
    background: rgba(156, 29, 58, 0.12) !important;
    color: #9C1D3A !important;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.5px;
    border: 1px solid rgba(156, 29, 58, 0.28) !important;
}
.badge-MEDIUM {
    background: rgba(186, 106, 36, 0.12) !important;
    color: #BA6A24 !important;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.5px;
    border: 1px solid rgba(186, 106, 36, 0.28) !important;
}
.badge-LOW {
    background: rgba(45, 122, 88, 0.12) !important;
    color: #2D7A58 !important;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.5px;
    border: 1px solid rgba(45, 122, 88, 0.28) !important;
}

/* Recommendation Cards */
.rec-card {
    background: linear-gradient(145deg, rgba(255, 255, 255, 0.95) 0%, rgba(252, 248, 244, 0.92) 100%);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(114, 47, 55, 0.14);
    border-radius: 16px;
    padding: 24px;
    margin: 14px 0;
    box-shadow: 0 10px 25px -8px rgba(114, 47, 55, 0.08);
    transition: all 0.25s ease;
}

.rec-card:hover {
    border-color: rgba(114, 47, 55, 0.35);
    box-shadow: 0 16px 32px -10px rgba(114, 47, 55, 0.16);
}

/* Streamlit Button Overrides - Velvet Wine Gradient */
.stButton > button {
    background: linear-gradient(135deg, #6B1D2F 0%, #8B1E3F 50%, #A3485E 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(114, 47, 55, 0.3) !important;
    border-radius: 12px !important;
    padding: 0.65rem 1.6rem !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 6px 18px rgba(114, 47, 55, 0.25) !important;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #581825 0%, #722F37 50%, #8B1E3F 100%) !important;
    box-shadow: 0 8px 24px rgba(114, 47, 55, 0.38) !important;
    transform: translateY(-2px) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
}

/* Form Inputs, Selectboxes, Textareas */
.stTextArea > div > textarea, .stTextInput > div > input {
    background: rgba(255, 255, 255, 0.95) !important;
    border: 1px solid rgba(114, 47, 55, 0.18) !important;
    color: #33181E !important;
    border-radius: 12px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    padding: 12px !important;
    font-size: 14px !important;
    transition: all 0.2s ease !important;
}

.stTextArea > div > textarea:focus, .stTextInput > div > input:focus {
    border-color: #722F37 !important;
    box-shadow: 0 0 0 3px rgba(114, 47, 55, 0.18) !important;
}

.stSelectbox > div > div {
    background: rgba(255, 255, 255, 0.95) !important;
    border: 1px solid rgba(114, 47, 55, 0.18) !important;
    color: #33181E !important;
    border-radius: 12px !important;
}

/* Tabs Styling - Floating Velvet Wine Segmented Controls */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(245, 237, 230, 0.85) !important;
    border: 1px solid rgba(114, 47, 55, 0.14) !important;
    border-radius: 14px !important;
    padding: 5px !important;
    gap: 6px !important;
}

.stTabs [data-baseweb="tab"] {
    color: #8C6C72 !important;
    border-radius: 10px !important;
    font-size: 13.5px !important;
    font-weight: 600 !important;
    padding: 8px 18px !important;
    transition: all 0.2s ease !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #722F37 0%, #8B1E3F 100%) !important;
    border: 1px solid rgba(114, 47, 55, 0.4) !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 14px rgba(114, 47, 55, 0.25) !important;
}

/* Plotly & Dataframe Rounded Wrappers */
.js-plotly-plot {
    border-radius: 16px !important;
    overflow: hidden !important;
    background: rgba(255, 255, 255, 0.9) !important;
    border: 1px solid rgba(114, 47, 55, 0.12) !important;
    box-shadow: 0 10px 25px -8px rgba(114, 47, 55, 0.08) !important;
}

[data-testid="stDataFrame"] {
    border-radius: 14px !important;
    overflow: hidden !important;
    border: 1px solid rgba(114, 47, 55, 0.12) !important;
    background: rgba(255, 255, 255, 0.9) !important;
    box-shadow: 0 10px 25px -8px rgba(114, 47, 55, 0.08) !important;
}

/* Expander Styling */
.streamlit-expanderHeader {
    background: rgba(255, 255, 255, 0.85) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(114, 47, 55, 0.12) !important;
    color: #380E16 !important;
    font-weight: 600 !important;
}

/* Loading Animation */
.loading-text {
    color: #722F37;
    font-weight: 700;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    animation: pulse 1.5s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.45; }
}
</style>
""", unsafe_allow_html=True)


# ── Session State ─────────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "analysis_results": None,
        "analysis_done": False,
        "feedback_col": None,
        "date_col": None,
        "rating_col": None,
        "df_raw": None,
        "single_result": None,
        "single_text": "",
        "groq_api_key": None,
        "activity_log": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="brand-container">
        <div class="brand-badge">⚡ AUTONOMOUS AI</div>
        <div class="brand-logo">Feedback.AI</div>
        <div class="brand-sub">Customer Intelligence Agent</div>
    </div>
    """, unsafe_allow_html=True)

    pages = {
        "🏠 Dashboard": "Dashboard",
        "🔍 Analyze Feedback": "Analyze Feedback",
        "📂 Upload Dataset": "Upload Dataset",
        "⚠️ Issues": "Issues",
        "💡 Insights": "Insights",
        "🎯 Recommendations": "Recommendations",
        "🤖 Agent Activity": "Agent Activity",
        "📊 Reports": "Reports",
        "🔎 Feedback Explorer": "Feedback Explorer",
    }

    page_labels = list(pages.keys())
    val_to_label = {v: k for k, v in pages.items()}

    if "_nav" in st.session_state:
        target_val = st.session_state.pop("_nav")
        if target_val in val_to_label:
            st.session_state["nav_selection"] = val_to_label[target_val]

    if "nav_selection" not in st.session_state:
        st.session_state["nav_selection"] = page_labels[0]

    selected = st.radio("Navigation", page_labels, key="nav_selection", label_visibility="collapsed")
    page = pages[selected]

    st.markdown("<div style='margin: 16px 0; border-top: 1px solid rgba(255,255,255,0.07);'></div>", unsafe_allow_html=True)

    # Status indicator
    if st.session_state.analysis_done:
        df = st.session_state.analysis_results.get("df")
        count = len(df) if df is not None else 0
        st.markdown(f"""
        <div class="status-pill complete">
            <div class="status-dot complete"></div>
            <div>
                <div style="color: #2D7A58; font-weight: 700; font-size: 12px; letter-spacing: 0.3px;">PIPELINE ACTIVE</div>
                <div style="color: #6E4D53; font-size: 11px; margin-top: 2px;">{count:,} records ingested</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-pill ready">
            <div class="status-dot ready"></div>
            <div>
                <div style="color: #722F37; font-weight: 700; font-size: 12px; letter-spacing: 0.3px;">SYSTEM READY</div>
                <div style="color: #6E4D53; font-size: 11px; margin-top: 2px;">Upload dataset or load sample</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Groq LLM Status & Key Management
    groq_active = is_groq_available(st.session_state.get("groq_api_key"))
    if groq_active:
        st.markdown(f"""
        <div style="background: rgba(114, 47, 55, 0.08); border: 1px solid rgba(114, 47, 55, 0.22);
                    border-radius: 12px; padding: 10px 14px; margin-top: 10px;">
            <div style="color: #722F37; font-weight: 700; font-size: 12px;">⚡ GROQ LLM ACTIVE</div>
            <div style="color: #6E4D53; font-size: 11px; margin-top: 2px;">{DEFAULT_MODEL}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        with st.expander("⚡ Connect Groq LLM", expanded=False):
            st.markdown("<div style='font-size:11.5px; color:#6E4D53; margin-bottom:8px;'>Add your free Groq key or save it in <code>.env</code> as <code>GROQ_API_KEY</code></div>", unsafe_allow_html=True)
            groq_input = st.text_input("Groq API Key:", type="password", placeholder="gsk_...", key="groq_key_input")
            if groq_input:
                st.session_state["groq_api_key"] = groq_input.strip()
                os.environ["GROQ_API_KEY"] = groq_input.strip()
                st.success("✅ Groq Connected!")
                st.rerun()

    st.markdown("""
    <div style="position: fixed; bottom: 20px; left: 18px; right: 18px; font-size: 11px; color: #8C6C72; line-height: 1.5;">
        <strong style="color: #722F37;">Customer Feedback Agent</strong><br>
        Autonomous Multi-Agent AI System
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER: render KPI cards
# ═══════════════════════════════════════════════════════════════════════════════
def kpi_card(title: str, value: str, sub: str = "", color: str = "#380E16", accent_gradient: str = "linear-gradient(90deg, #722F37, #A3485E)"):
    st.markdown(f"""
    <div class="kpi-card">
        <div style="position: absolute; top: 0; left: 0; right: 0; height: 3px; background: {accent_gradient};"></div>
        <div class="kpi-title">{title}</div>
        <div class="kpi-value" style="color:{color};">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def priority_badge(priority: str) -> str:
    return f'<span class="badge-{priority}">{priority}</span>'


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD / HOME
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Dashboard":
    # Luxury Hero
    st.markdown("""
    <div class="hero-wrapper">
        <div class="hero-chip">
            <span class="status-dot ready" style="margin-right: 4px;"></span>
            Enterprise Autonomous AI • Multi-Stage Intelligence Pipeline
        </div>
        <h1 class="hero-title">
            Customer Feedback <span class="gradient-text">Analysis Agent</span>
        </h1>
        <p class="hero-desc">
            Transform raw, unstructured customer streams into clustered recurring issues, sentiment trajectories, urgency priority scores, and strategic executive roadmaps in real-time.
        </p>
        <div class="hero-stats-row">
            <div class="hero-stat-badge">
                <span style="color:#2D7A58;">●</span> 6-Stage Agentic Pipeline
            </div>
            <div class="hero-stat-badge">
                <span style="color:#722F37;">●</span> Semantic DBSCAN & TF-IDF Clustering
            </div>
            <div class="hero-stat-badge">
                <span style="color:#A3485E;">●</span> Multi-Factor Priority Matrix
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick action buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("🔍 Analyze Feedback", use_container_width=True):
            st.session_state["_nav"] = "Analyze Feedback"
            st.rerun()
    with col2:
        if st.button("📂 Upload Dataset", use_container_width=True):
            st.session_state["_nav"] = "Upload Dataset"
            st.rerun()

    # Load sample data button
    sample_path = ROOT / "data" / "sample_feedback.csv"
    if sample_path.exists():
        with col3:
            st.markdown("")
            if st.button("⚡ Load Sample Dataset & Run Analysis", use_container_width=True):
                df_sample = pd.read_csv(sample_path)
                fb_col = detect_feedback_column(df_sample)
                date_col = detect_date_column(df_sample)
                rating_col = detect_rating_column(df_sample)

                with st.spinner("🤖 Agent is analyzing sample data..."):
                    agent = CustomerFeedbackAgent()
                    results = agent.analyze_dataset(df_sample, fb_col, date_col, rating_col)

                st.session_state.analysis_results = results
                st.session_state.analysis_done = True
                st.session_state.feedback_col = fb_col
                st.session_state.date_col = date_col
                st.session_state.rating_col = rating_col
                st.session_state.df_raw = df_sample
                st.success("✅ Analysis complete!")
                st.rerun()

    st.markdown("<div style='margin: 20px 0; border-top: 1px solid rgba(114,47,55,0.12);'></div>", unsafe_allow_html=True)

    # KPI cards
    if st.session_state.analysis_done:
        results = st.session_state.analysis_results
        df = results.get("df", pd.DataFrame())
        priority_dist = results.get("priority_distribution", {})

        st.markdown('<div class="section-header">📊 Analysis Summary</div>', unsafe_allow_html=True)

        total = len(df)

        def _safe_count(col_name: str, val: str) -> int:
            if col_name not in df.columns:
                return 0
            series = df[col_name]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, -1]
            cnt = (series == val).sum()
            if isinstance(cnt, pd.Series):
                return int(cnt.iloc[0])
            return int(cnt)

        pos = _safe_count("sentiment", "Positive")
        neutral = _safe_count("sentiment", "Neutral")
        neg = _safe_count("sentiment", "Negative")
        high_p = priority_dist.get("HIGH", 0)
        if isinstance(high_p, pd.Series):
            high_p = int(high_p.iloc[0])
        else:
            try:
                high_p = int(high_p)
            except Exception:
                high_p = 0
        recurring_count = len(results.get("recurring_issues", []))

        pos_pct = round(pos / total * 100, 1) if total else 0.0
        neu_pct = round(neutral / total * 100, 1) if total else 0.0
        neg_pct = round(neg / total * 100, 1) if total else 0.0

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1: kpi_card("Total Feedback", f"{total:,}", "records analyzed", "#380E16", "linear-gradient(90deg, #722F37, #A3485E)")
        with c2: kpi_card("Positive", f"{pos:,}", f"{pos_pct}% share", "#2D7A58", "linear-gradient(90deg, #2D7A58, #459B73)")
        with c3: kpi_card("Neutral", f"{neutral:,}", f"{neu_pct}% share", "#7D656A", "linear-gradient(90deg, #7D656A, #A1888E)")
        with c4: kpi_card("Negative", f"{neg:,}", f"{neg_pct}% share", "#9C1D3A", "linear-gradient(90deg, #9C1D3A, #C4435F)")
        with c5: kpi_card("High Priority", f"{high_p:,}", "critical focus", "#8B1430", "linear-gradient(90deg, #8B1430, #B32D4B)")
        with c6: kpi_card("Recurring Issues", f"{recurring_count:,}", "semantic clusters", "#BA6A24", "linear-gradient(90deg, #BA6A24, #D98845)")

        st.markdown("")

        # Charts Row 1
        ch1, ch2 = st.columns(2)
        with ch1:
            fig = sentiment_pie_chart(df)
            if fig: st.plotly_chart(fig, use_container_width=True)
        with ch2:
            fig = emotion_bar_chart(df)
            if fig: st.plotly_chart(fig, use_container_width=True)

        # Charts Row 2
        ch3, ch4 = st.columns(2)
        with ch3:
            fig = topic_bar_chart(df)
            if fig: st.plotly_chart(fig, use_container_width=True)
        with ch4:
            if priority_dist:
                fig = priority_donut_chart(priority_dist)
                st.plotly_chart(fig, use_container_width=True)

        # Charts Row 3 — Trends (only if date available)
        trend_data = results.get("trend_data", {})
        if trend_data.get("date_available"):
            ch5, ch6 = st.columns(2)
            with ch5:
                fig = sentiment_trend_chart(trend_data.get("sentiment_trend"))
                if fig: st.plotly_chart(fig, use_container_width=True)
            with ch6:
                fig = volume_trend_chart(trend_data.get("volume_trend"))
                if fig: st.plotly_chart(fig, use_container_width=True)

        # Recurring Issues Chart
        recurring = results.get("recurring_issues", [])
        if recurring:
            st.markdown('<div class="section-header">🔁 Recurring Issues</div>', unsafe_allow_html=True)
            fig = recurring_issues_chart(recurring)
            if fig: st.plotly_chart(fig, use_container_width=True)

    else:
        st.markdown('<div class="section-header">📂 Ingest Feedback Dataset</div>', unsafe_allow_html=True)
        quick_file = st.file_uploader(
            "Drag & drop your CSV or Excel file here:",
            type=["csv", "xlsx", "xls"],
            key="dash_quick_uploader",
            help="Supports CSV, Excel (.xlsx, .xls), and various encodings."
        )
        if quick_file:
            try:
                if quick_file.name.lower().endswith((".csv", ".txt")):
                    raw_bytes = quick_file.getvalue()
                    df_quick = None
                    for enc in ["utf-8", "utf-8-sig", "latin1", "cp1252", "iso-8859-1"]:
                        for sep in [None, ",", ";", "\t", "|"]:
                            try:
                                import io
                                df_quick = pd.read_csv(
                                    io.BytesIO(raw_bytes),
                                    encoding=enc,
                                    sep=sep,
                                    engine="python" if sep is not None else None,
                                    on_bad_lines="skip"
                                )
                                if df_quick is not None and len(df_quick.columns) >= 1 and len(df_quick) > 0:
                                    break
                            except Exception:
                                continue
                        if df_quick is not None and len(df_quick.columns) >= 1 and len(df_quick) > 0:
                            break
                    if df_quick is None:
                        import io
                        df_quick = pd.read_csv(io.BytesIO(raw_bytes), encoding_errors="replace", on_bad_lines="skip")
                else:
                    df_quick = pd.read_excel(quick_file)

                df_quick.columns = [str(c).strip() for c in df_quick.columns]
                st.success(f"✅ Loaded **{quick_file.name}**: {len(df_quick):,} rows, {len(df_quick.columns)} columns")

                fb_col = detect_feedback_column(df_quick)
                if not fb_col and len(df_quick.columns) > 0:
                    fb_col = df_quick.columns[0]
                dt_col = detect_date_column(df_quick)
                rt_col = detect_rating_column(df_quick)

                c_q1, c_q2 = st.columns([2, 1])
                with c_q1:
                    st.markdown(f"<div style='font-size:13.5px; color:#6E4D53; padding-top:6px;'>Selected text column: <strong>`{fb_col}`</strong> &nbsp;|&nbsp; Records: <strong>{len(df_quick):,}</strong></div>", unsafe_allow_html=True)
                with c_q2:
                    if st.button("🚀 Analyze Uploaded Dataset", key="btn_quick_analyze", use_container_width=True):
                        with st.spinner("🤖 Multi-agent pipeline executing analysis..."):
                            agent = CustomerFeedbackAgent()
                            results = agent.analyze_dataset(df_quick, fb_col, dt_col, rt_col)
                        st.session_state.analysis_results = results
                        st.session_state.analysis_done = True
                        st.session_state.feedback_col = fb_col
                        st.session_state.date_col = dt_col
                        st.session_state.rating_col = rt_col
                        st.session_state.df_raw = df_quick
                        st.success("✅ Analysis complete!")
                        st.rerun()
            except Exception as e:
                st.error(f"❌ Error loading file: {e}")

        # Feature cards
        st.markdown('<div class="section-header">🚀 Features</div>', unsafe_allow_html=True)
        features = [
            ("🧠", "Sentiment Intelligence", "Classify feedback as Positive, Neutral, or Negative with confidence scores."),
            ("🔍", "Issue Detection", "Automatically identify complaints and operational issues in feedback."),
            ("❤️", "Customer Emotion", "Detect emotions: Happy, Frustrated, Angry, Disappointed, Confused."),
            ("🔁", "Recurring Problems", "Cluster similar complaints using semantic similarity."),
            ("🚨", "Priority Analysis", "Score issues by urgency using multi-factor priority engine."),
            ("🎯", "AI Recommendations", "Convert findings into actionable business recommendations."),
        ]
        cols = st.columns(3)
        for i, (icon, title, desc) in enumerate(features):
            with cols[i % 3]:
                st.markdown(f"""
                <div class="feature-card">
                    <div class="feature-icon">{icon}</div>
                    <div class="feature-title">{title}</div>
                    <div class="feature-desc">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("""
        <div style="text-align: center; margin-top: 30px; padding: 32px;
                    background: linear-gradient(145deg, rgba(255, 255, 255, 0.95) 0%, rgba(250, 244, 238, 0.95) 100%);
                    border-radius: 16px; border: 1px dashed rgba(114, 47, 55, 0.28); box-shadow: 0 10px 25px -5px rgba(114, 47, 55, 0.06);">
            <div style="font-family: 'Outfit', sans-serif; font-size: 20px; font-weight: 700; color: #380E16; margin-bottom: 8px;">
                Ready to Unlock Customer Intelligence?
            </div>
            <div style="font-size: 14px; color: #6E4D53; max-width: 500px; margin: 0 auto;">
                Upload your feedback dataset above in CSV/Excel format or click "Load Sample Dataset" to run the 6-stage autonomous pipeline.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYZE FEEDBACK (Single)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Analyze Feedback":
    st.markdown('<div class="section-header">🔍 Single Feedback Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<p style="color: #6E4D53;">Enter any customer feedback to get instant AI analysis.</p>',
                unsafe_allow_html=True)

    examples = [
        "The product arrived late and customer support did not respond to my complaint.",
        "I love this product! The quality is excellent and delivery was super fast.",
        "My payment was deducted but the order was cancelled without any refund.",
        "The app crashes every time I try to checkout. Very frustrating experience.",
        "Received a completely wrong item. Ordered headphones, got a charger.",
        "Product quality is decent but the price is a bit high.",
    ]

    col_ex, col_btn = st.columns([3, 1])
    with col_ex:
        example_choice = st.selectbox("📋 Try an example:", ["Custom input"] + examples)
    
    default_text = "" if example_choice == "Custom input" else example_choice
    feedback_input = st.text_area(
        "Customer Feedback Text:",
        value=default_text,
        height=120,
        placeholder="Type or paste customer feedback here...",
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        analyze_btn = st.button("🤖 Analyze Feedback", use_container_width=True)

    if analyze_btn:
        if not feedback_input.strip():
            st.warning("⚠️ Please enter some feedback text.")
        else:
            with st.spinner("🤖 Agent analyzing feedback..."):
                agent = CustomerFeedbackAgent()
                result = agent.analyze_single(feedback_input.strip())
                st.session_state.single_result = result
                st.session_state.single_text = feedback_input.strip()

    if st.session_state.single_result:
        result = st.session_state.single_result
        st.markdown("<div style='margin: 20px 0; border-top: 1px solid rgba(114, 47, 55, 0.15);'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">📊 Analysis Results</div>', unsafe_allow_html=True)

        # Sentiment card (prominent)
        sentiment = result.get("sentiment", "Neutral")
        sent_color = SENTIMENT_COLORS.get(sentiment, "#8C6C72")
        sentiment_conf = result.get("sentiment_confidence", 0)
        rgb_val = '45,122,88' if sentiment=='Positive' else '156,29,58' if sentiment=='Negative' else '140,108,114'

        st.markdown(f"""
        <div style="background: linear-gradient(145deg, rgba(255, 255, 255, 0.96) 0%, rgba(252, 248, 244, 0.95) 100%);
                    border: 1.5px solid rgba({rgb_val}, 0.35);
                    box-shadow: 0 12px 30px -5px rgba(114, 47, 55, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.8);
                    border-radius: 18px; padding: 28px 20px; text-align: center; margin: 18px 0;">
            <div style="font-size: 11px; color: #8C6C72; text-transform: uppercase; letter-spacing: 2px; font-weight: 700; margin-bottom: 8px;">
                Detected Sentiment Polarity
            </div>
            <div style="font-family: 'Outfit', sans-serif; font-size: 40px; font-weight: 900; color: {sent_color}; margin-bottom: 8px; letter-spacing: -0.5px;">
                {sentiment.upper()}
            </div>
            <div style="font-size: 13px; color: #6E4D53; font-weight: 500;">
                Model Confidence: <span style="color: {sent_color}; font-weight: 700;">{sentiment_conf}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Detail cards
        c1, c2, c3 = st.columns(3)
        with c1:
            emotion = result.get("emotion", "Neutral")
            emotion_color = EMOTION_COLORS.get(emotion, "#722F37")
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Emotion</div>
                <div class="kpi-value" style="font-size:22px; color:{emotion_color};">{emotion}</div>
                <div class="kpi-sub">Confidence: {result.get('emotion_confidence', 0)}%</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            topic = result.get("topic", "Other")
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Topic</div>
                <div class="kpi-value" style="font-size:20px; color:#722F37;">{topic}</div>
                <div class="kpi-sub">Confidence: {result.get('topic_confidence', 0)}%</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            priority = result.get("priority", "LOW")
            p_color = PRIORITY_COLORS.get(priority, "#8C6C72")
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Priority</div>
                <div class="kpi-value" style="font-size:22px; color:{p_color};">{priority}</div>
                <div class="kpi-sub">Score: {result.get('priority_score', 0)}/12</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")

        # Issue + Action
        c4, c5 = st.columns(2)
        with c4:
            issue = result.get("detected_issue", "General Feedback")
            is_complaint = result.get("is_complaint", False)
            st.markdown(f"""
            <div style="background: linear-gradient(145deg, rgba(255, 255, 255, 0.95) 0%, rgba(252, 248, 244, 0.92) 100%);
                        border: 1px solid rgba(114, 47, 55, 0.14); border-radius: 14px; padding: 20px;
                        box-shadow: 0 8px 20px rgba(114, 47, 55, 0.06);">
                <div style="font-size: 11px; color: #8C6C72; text-transform: uppercase; letter-spacing: 1px; font-weight: 700; margin-bottom: 8px;">
                    Detected Issue Category
                </div>
                <div style="font-family: 'Outfit', sans-serif; font-size: 19px; font-weight: 700; color: #380E16; margin-bottom: 8px;">
                    {issue}
                </div>
                <div style="font-size: 12.5px; color: #6E4D53;">
                    Confidence: <strong style="color: #380E16;">{result.get('issue_confidence', 0)}%</strong>
                    &nbsp;|&nbsp;
                    {'<span style="color:#9C1D3A; font-weight:600;">⚠️ Complaint Detected</span>' if is_complaint else '<span style="color:#2D7A58; font-weight:600;">✅ Non-Complaint Feedback</span>'}
                </div>
            </div>
            """, unsafe_allow_html=True)
        with c5:
            action = result.get("recommended_action", "Monitor and review")
            details = result.get("recommendation_details", "")
            st.markdown(f"""
            <div style="background: linear-gradient(145deg, rgba(255, 255, 255, 0.95) 0%, rgba(245, 250, 247, 0.92) 100%);
                        border: 1px solid rgba(45, 122, 88, 0.25); border-radius: 14px; padding: 20px;
                        box-shadow: 0 8px 20px rgba(45, 122, 88, 0.06);">
                <div style="font-size: 11px; color: #2D7A58; text-transform: uppercase; letter-spacing: 1px; font-weight: 700; margin-bottom: 8px;">
                    Prescribed Action
                </div>
                <div style="font-family: 'Outfit', sans-serif; font-size: 18px; font-weight: 700; color: #2D7A58; margin-bottom: 8px;">
                    {action}
                </div>
                <div style="font-size: 13px; color: #6E4D53; line-height: 1.5;">
                    {details[:140]}...
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Priority reasons
        reasons = result.get("priority_reasons", [])
        if reasons:
            st.markdown(f"""
            <div style="background: linear-gradient(145deg, rgba(255, 255, 255, 0.95) 0%, rgba(255, 250, 240, 0.92) 100%);
                        border: 1px solid rgba(186, 106, 36, 0.25); border-radius: 14px; padding: 20px; margin-top: 14px;
                        box-shadow: 0 8px 20px rgba(186, 106, 36, 0.06);">
                <div style="font-size: 11px; color: #BA6A24; text-transform: uppercase; font-weight: 700; letter-spacing: 1px; margin-bottom: 12px;">
                    Multi-Factor Urgency Breakdown ({priority} Priority)
                </div>
                {''.join(f'<div style="color: #380E16; font-size: 13.5px; margin: 6px 0; display: flex; align-items: center; gap: 8px;"><span style="color:#BA6A24;">▸</span> {r}</div>' for r in reasons)}
            </div>
            """, unsafe_allow_html=True)

        # Optional Groq LLM Root-Cause Synthesis
        groq_key = st.session_state.get("groq_api_key") or os.getenv("GROQ_API_KEY")
        if is_groq_available(groq_key):
            st.markdown("")
            with st.expander(f"⚡ Groq AI ({DEFAULT_MODEL}) — Executive Root-Cause Synthesis", expanded=True):
                with st.spinner(f"🤖 Groq {DEFAULT_MODEL} synthesizing root-cause & resolution..."):
                    groq_response = explain_feedback_with_groq(
                        feedback_text=st.session_state.get("single_text", ""),
                        sentiment=sentiment,
                        topic=result.get("topic", "Other"),
                        issue=result.get("detected_issue", "General Feedback"),
                        priority=priority,
                        api_key=groq_key
                    )
                if groq_response:
                    st.markdown(f"""
                    <div style="background: rgba(114, 47, 55, 0.04); border: 1px solid rgba(114, 47, 55, 0.18);
                                border-radius: 12px; padding: 18px 22px; color: #380E16; line-height: 1.6; font-size: 14px;">
                        <div style="font-size: 11px; color: #722F37; text-transform: uppercase; font-weight: 700; letter-spacing: 1px; margin-bottom: 8px;">
                            ⚡ {DEFAULT_MODEL} Executive Debrief
                        </div>
                        {groq_response.replace(chr(10), '<br>')}
                    </div>
                    """, unsafe_allow_html=True)



# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: UPLOAD DATASET
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Upload Dataset":
    st.markdown('<div class="section-header">📂 Upload Customer Feedback Dataset</div>', unsafe_allow_html=True)
    st.markdown('<p style="color: #6E4D53;">Upload a CSV or Excel file containing customer feedback.</p>',
                unsafe_allow_html=True)

    # Format guide
    with st.expander("📋 Supported File Format"):
        st.markdown("""
        **Required columns** (at least one):
        - `feedback`, `review`, `comment`, `customer_feedback`, `review_text`, `complaint`
        
        **Optional columns** (auto-detected):
        - `date` / `created_at` — enables trend analysis
        - `rating` / `score` — enables rating analysis  
        - `id`, `customer_id`, `product`, `category`, `channel`
        
        **Formats**: CSV (.csv) or Excel (.xlsx, .xls)
        """)

    uploaded_file = st.file_uploader(
        "Drag and drop your file here",
        type=["csv", "xlsx", "xls"],
        help="Maximum file size: 200MB",
    )

    if uploaded_file:
        try:
            if uploaded_file.name.lower().endswith((".csv", ".txt")):
                raw_bytes = uploaded_file.getvalue()
                df_raw = None
                # Try multiple encodings and separators
                for enc in ["utf-8", "utf-8-sig", "latin1", "cp1252", "iso-8859-1"]:
                    for sep in [None, ",", ";", "\t", "|"]:
                        try:
                            import io
                            df_raw = pd.read_csv(
                                io.BytesIO(raw_bytes),
                                encoding=enc,
                                sep=sep,
                                engine="python" if sep is not None else None,
                                on_bad_lines="skip"
                            )
                            if df_raw is not None and len(df_raw.columns) >= 1 and len(df_raw) > 0:
                                break
                        except Exception:
                            continue
                    if df_raw is not None and len(df_raw.columns) >= 1 and len(df_raw) > 0:
                        break

                if df_raw is None:
                    import io
                    df_raw = pd.read_csv(io.BytesIO(raw_bytes), encoding_errors="replace", on_bad_lines="skip")
            else:
                df_raw = pd.read_excel(uploaded_file)

            # Clean and normalize column names
            df_raw.columns = [str(c).strip() for c in df_raw.columns]

            st.success(f"✅ File loaded: **{uploaded_file.name}** — {len(df_raw):,} rows, {len(df_raw.columns)} columns")

            # Show preview
            st.markdown('<div class="section-header">📋 Data Preview</div>', unsafe_allow_html=True)
            st.dataframe(df_raw.head(5), use_container_width=True)

            # Column detection
            detected_fb = detect_feedback_column(df_raw)
            detected_date = detect_date_column(df_raw)
            detected_rating = detect_rating_column(df_raw)

            st.markdown('<div class="section-header">⚙️ Column Configuration</div>', unsafe_allow_html=True)

            cols_list = df_raw.columns.tolist()
            default_fb_idx = cols_list.index(detected_fb) if detected_fb and detected_fb in cols_list else 0

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                feedback_col = st.selectbox(
                    "📝 Feedback Column *",
                    options=cols_list,
                    index=default_fb_idx,
                    help="Column containing the customer feedback text"
                )
            with col_b:
                date_options = ["None"] + cols_list
                date_default = date_options.index(detected_date) if detected_date in date_options else 0
                date_col = st.selectbox(
                    "📅 Date Column (optional)",
                    options=date_options,
                    index=date_default,
                    help="Enables trend analysis over time"
                )
                date_col = None if date_col == "None" else date_col
            with col_c:
                rating_options = ["None"] + cols_list
                rating_default = rating_options.index(detected_rating) if detected_rating in rating_options else 0
                rating_col = st.selectbox(
                    "⭐ Rating Column (optional)",
                    options=rating_options,
                    index=rating_default,
                    help="Enables rating analysis"
                )
                rating_col = None if rating_col == "None" else rating_col

            # Data quality preview
            non_null = df_raw[feedback_col].notna().sum() if feedback_col in df_raw.columns else 0
            st.info(f"📊 **{non_null:,}** non-null feedback records found in column `{feedback_col}`")

            c_btn1, c_btn2 = st.columns([1, 1])
            with c_btn1:
                start_btn = st.button("🚀 Start Autonomous Analysis", use_container_width=True)

            if start_btn:
                progress_bar = st.progress(0)
                status_text = st.empty()

                def progress_cb(step, total, message):
                    progress_bar.progress(step / total)
                    status_text.markdown(f'<span class="loading-text">🤖 {message}</span>',
                                         unsafe_allow_html=True)

                with st.spinner(""):
                    agent = CustomerFeedbackAgent(progress_callback=progress_cb)
                    results = agent.analyze_dataset(df_raw, feedback_col, date_col, rating_col)

                progress_bar.progress(1.0)
                status_text.empty()

                st.session_state.analysis_results = results
                st.session_state.analysis_done = True
                st.session_state.feedback_col = feedback_col
                st.session_state.date_col = date_col
                st.session_state.rating_col = rating_col
                st.session_state.df_raw = df_raw

                st.success("✅ Analysis complete! All 6 analytical stages executed.")
                st.session_state["_nav"] = "Dashboard"
                st.rerun()

        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
            st.info("💡 Tip: Ensure your file contains at least one column with customer feedback text.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ISSUES
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Issues":
    st.markdown('<div class="section-header">⚠️ Detected Issues</div>', unsafe_allow_html=True)

    if not st.session_state.analysis_done:
        st.info("📂 Upload a dataset or load sample data to view issues.")
    else:
        results = st.session_state.analysis_results
        df = results.get("df", pd.DataFrame())
        issue_summary = results.get("issue_summary", pd.DataFrame())
        recurring = results.get("recurring_issues", [])

        tab1, tab2 = st.tabs(["📋 Issue Summary", "🔁 Recurring Issues"])

        with tab1:
            if issue_summary is not None and len(issue_summary) > 0:
                # Priority filter
                priority_filter = st.multiselect(
                    "Filter by priority:",
                    ["HIGH", "MEDIUM", "LOW"],
                    default=["HIGH", "MEDIUM", "LOW"]
                )

                # Add priority column to issue summary
                if "priority" not in issue_summary.columns and "detected_issue" in df.columns:
                    priority_map = (
                        df.groupby("detected_issue")["priority"].agg(
                            lambda x: x.value_counts().index[0] if len(x) > 0 else "LOW"
                        ).to_dict() if "priority" in df.columns else {}
                    )
                    issue_summary["priority"] = issue_summary["detected_issue"].map(priority_map).fillna("LOW")

                for _, row in issue_summary.iterrows():
                    priority = row.get("priority", "LOW")
                    if priority not in priority_filter:
                        continue
                    p_color = PRIORITY_COLORS.get(priority, "#8C6C72")
                    st.markdown(f"""
                    <div style="background: linear-gradient(145deg, rgba(255, 255, 255, 0.95) 0%, rgba(252, 248, 244, 0.92) 100%);
                                border: 1px solid rgba(114, 47, 55, 0.12);
                                border-left: 4px solid {p_color}; border-radius: 14px;
                                padding: 18px 22px; margin: 10px 0;
                                box-shadow: 0 10px 25px -8px rgba(114, 47, 55, 0.08);">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <div style="font-family: 'Outfit', sans-serif; font-size: 18px; font-weight: 700; color: #380E16; margin-bottom: 6px;">
                                    {row.get('detected_issue', 'Unknown')}
                                </div>
                                <div style="font-size: 13px; color: #6E4D53;">
                                    Occurrences: <strong style="color:#380E16;">{int(row.get('occurrences', 0))}</strong>
                                    &nbsp;|&nbsp;
                                    Share: <strong style="color:#380E16;">{row.get('percentage', 0)}%</strong>
                                    &nbsp;|&nbsp;
                                    Avg Sentiment: <strong style="color:#380E16;">{row.get('avg_sentiment', 'N/A')}</strong>
                                </div>
                            </div>
                            <div>
                                <span class="badge-{priority}">{priority}</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No specific issue patterns detected.")

        with tab2:
            if recurring:
                st.markdown(f"**{len(recurring)} recurring issue clusters detected** through semantic analysis")
                st.markdown("")

                fig = recurring_issues_chart(recurring)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown("")
                for i, issue in enumerate(recurring):
                    p_color = PRIORITY_COLORS.get(issue["priority"], "#8C6C72")
                    with st.expander(f"🔁 {issue['issue_label']} — {issue['occurrences']} occurrences"):
                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            st.metric("Occurrences", issue["occurrences"])
                        with c2:
                            st.metric("Share", f"{issue['percentage']}%")
                        with c3:
                            st.metric("Avg Sentiment", issue["avg_sentiment"])
                        with c4:
                            st.metric("Priority", issue["priority"])

                        st.markdown("**Sample Customer Feedback:**")
                        for sample in issue.get("sample_feedbacks", []):
                            st.markdown(f'<div style="background: rgba(255, 255, 255, 0.85); border: 1px solid rgba(114, 47, 55, 0.12); '
                                        f'border-radius: 8px; padding: 12px 14px; margin: 6px 0; color: #4A1521; font-size: 13px; '
                                        f'font-style: italic;">"{sample}"</div>', unsafe_allow_html=True)
            else:
                st.info("No significant recurring issue clusters detected. Try uploading a larger dataset.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Insights":
    st.markdown('<div class="section-header">💡 AI-Generated Insights</div>', unsafe_allow_html=True)
    st.markdown('<p style="color: #6E4D53;">All insights are dynamically synthesized from your actual feedback telemetry.</p>',
                unsafe_allow_html=True)

    if not st.session_state.analysis_done:
        st.info("📂 Upload a dataset or load sample data to view insights.")
    else:
        insights = st.session_state.analysis_results.get("insights", [])

        if not insights:
            st.warning("No insights generated. Please analyze a dataset first.")
        else:
            severity_filter = st.multiselect(
                "Filter by severity:",
                ["critical", "warning", "info"],
                default=["critical", "warning", "info"]
            )

            filtered = [i for i in insights if i["severity"] in severity_filter]

            for insight in filtered:
                severity = insight["severity"]
                border_color = {"critical": "#9C1D3A", "warning": "#BA6A24", "info": "#722F37"}.get(severity, "#8C6C72")
                bg = {"critical": "rgba(156, 29, 58, 0.06)", "warning": "rgba(186, 106, 36, 0.06)",
                      "info": "rgba(114, 47, 55, 0.06)"}.get(severity, "rgba(255, 255, 255, 0.9)")

                st.markdown(f"""
                <div style="background: {bg}; border: 1px solid rgba(114, 47, 55, 0.14);
                            border-left: 4px solid {border_color}; border-radius: 14px;
                            padding: 20px 22px; margin: 12px 0; box-shadow: 0 8px 20px rgba(114, 47, 55, 0.06);">
                    <div style="display: flex; align-items: center; margin-bottom: 8px;">
                        <span style="font-size: 22px; margin-right: 12px;">{insight['icon']}</span>
                        <span style="font-family: 'Outfit', sans-serif; font-size: 17px; font-weight: 700; color: #380E16;">{insight['title']}</span>
                        <span style="margin-left: auto; font-size: 11px; color: {border_color}; 
                                     text-transform: uppercase; letter-spacing: 1.5px; font-weight: 700;
                                     background: rgba(255,255,255,0.7); padding: 4px 10px; border-radius: 9999px; border: 1px solid rgba(114, 47, 55, 0.12);">
                            {severity}
                        </span>
                    </div>
                    <div style="font-size: 14px; color: #380E16; line-height: 1.6;">
                        {insight['description']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="text-align: center; margin-top: 20px; padding: 14px;
                        background: rgba(255, 255, 255, 0.85); border: 1px solid rgba(114, 47, 55, 0.12);
                        border-radius: 10px; color: #6E4D53; font-size: 13px;">
                {len(filtered)} insights displayed | Synthesized autonomously by Customer Feedback Agent
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Recommendations":
    st.markdown('<div class="section-header">🎯 Actionable Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<p style="color: #6E4D53;">Evidence-based strategic actions mapped directly to detected problem clusters.</p>',
                unsafe_allow_html=True)

    if not st.session_state.analysis_done:
        st.info("📂 Upload a dataset or load sample data to view recommendations.")
    else:
        recs = st.session_state.analysis_results.get("recommendations", [])

        if not recs:
            st.warning("No recommendations generated. Please analyze a dataset first.")
        else:
            priority_filter = st.multiselect(
                "Filter by priority:",
                ["HIGH", "MEDIUM", "LOW"],
                default=["HIGH", "MEDIUM"]
            )

            filtered = [r for r in recs if r["priority"] in priority_filter]

            for i, rec in enumerate(filtered, 1):
                priority = rec["priority"]
                p_color = PRIORITY_COLORS.get(priority, "#8C6C72")

                st.markdown(f"""
                <div class="rec-card">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px;">
                        <div>
                            <div style="font-family: 'Outfit', sans-serif; font-size: 19px; font-weight: 700; color: #380E16; margin-bottom: 4px;">
                                {i}. {rec['action']}
                            </div>
                            <div style="font-size: 13px; color: #6E4D53;">
                                Target Issue: <strong style="color: #722F37;">{rec['issue']}</strong>
                            </div>
                        </div>
                        <span class="badge-{priority}">{priority}</span>
                    </div>
                    
                    <div style="background: rgba(114, 47, 55, 0.05); border: 1px solid rgba(114, 47, 55, 0.18); border-radius: 10px; padding: 14px; margin: 12px 0;">
                        <div style="font-size: 11px; color: #722F37; margin-bottom: 4px; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;">Synthesized Evidence</div>
                        <div style="font-size: 13.5px; color: #380E16;">{rec['evidence']}</div>
                    </div>
                    
                    <div style="font-size: 14px; color: #4A1521; line-height: 1.6; margin: 12px 0;">
                        {rec['details']}
                    </div>
                    
                    <div style="display: flex; gap: 16px; margin-top: 14px; flex-wrap: wrap;">
                        <div style="background: rgba(45, 122, 88, 0.08); border: 1px solid rgba(45, 122, 88, 0.25); border-radius: 8px; padding: 8px 14px;">
                            <div style="font-size: 10.5px; color: #2D7A58; text-transform: uppercase; font-weight: 700;">KPI Target</div>
                            <div style="font-size: 13px; font-weight: 600; color: #2D7A58;">{rec['kpi']}</div>
                        </div>
                        <div style="background: rgba(114, 47, 55, 0.08); border: 1px solid rgba(114, 47, 55, 0.25); border-radius: 8px; padding: 8px 14px;">
                            <div style="font-size: 10.5px; color: #722F37; text-transform: uppercase; font-weight: 700;">Department</div>
                            <div style="font-size: 13px; font-weight: 600; color: #722F37;">{rec['department']}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: AGENT ACTIVITY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Agent Activity":
    st.markdown('<div class="section-header">🤖 Agent Activity Log</div>', unsafe_allow_html=True)
    st.markdown('<p style="color: #6E4D53;">The autonomous execution telemetry displaying every tool invocation and latency metric.</p>',
                unsafe_allow_html=True)

    if not st.session_state.analysis_done:
        st.info("📂 Upload a dataset or analyze feedback to inspect live agent activity.")

        # Show architecture diagram
        st.markdown('<div class="section-header">🏗️ Agent Architecture</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background: linear-gradient(145deg, rgba(255, 255, 255, 0.95) 0%, rgba(250, 244, 238, 0.95) 100%);
                    border: 1px solid rgba(114, 47, 55, 0.2); border-radius: 16px; padding: 28px;
                    font-family: 'JetBrains Mono', monospace; color: #380E16; box-shadow: 0 15px 35px -10px rgba(114, 47, 55, 0.08);">
            <div style="font-family: 'Outfit', sans-serif; color: #722F37; font-weight: 800; font-size: 18px; margin-bottom: 16px; letter-spacing: -0.3px;">
                CUSTOMER FEEDBACK ANALYSIS AGENT — MULTI-TOOL ARCHITECTURE
            </div>
            <div style="color: #A3485E;">│</div>
            <div>├── 🧹 Data Cleaning Tool (Sanitization, Emoji/Unicode, Stopword Pruning)</div>
            <div style="color: #A3485E;">│</div>
            <div>├── 💬 Sentiment Analysis Tool (Hybrid VADER + TextBlob Polarity)</div>
            <div style="color: #A3485E;">│</div>
            <div>├── ❤️ Emotion Analysis Tool (Emotion Vector Classification)</div>
            <div style="color: #A3485E;">│</div>
            <div>├── 🏷️ Topic Detection Tool (TF-IDF N-Gram Modeling)</div>
            <div style="color: #A3485E;">│</div>
            <div>├── ⚠️ Issue Detector Tool (Complaint Pattern Categorization)</div>
            <div style="color: #A3485E;">│</div>
            <div>├── 🔁 Recurring Issue Detector (DBSCAN Semantic Clustering)</div>
            <div style="color: #A3485E;">│</div>
            <div>├── 🚨 Priority Scoring Engine (5-Factor Urgency Matrix)</div>
            <div style="color: #A3485E;">│</div>
            <div>├── 📈 Temporal Trend Analyzer (Time-Series Moving Windows)</div>
            <div style="color: #A3485E;">│</div>
            <div>├── 💡 Insight Generator (Autonomous Telemetry Mining)</div>
            <div style="color: #A3485E;">│</div>
            <div>├── 🎯 Recommendation Engine (Prescriptive Action Mapping)</div>
            <div style="color: #A3485E;">│</div>
            <div>└── 📄 Report Generator (Executive PDF + Enriched Dataset Export)</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        activity_log = st.session_state.analysis_results.get("activity_log", [])

        # Summary stats
        total_steps = len(activity_log)
        total_time = sum(a.get("duration_ms", 0) for a in activity_log)
        errors = sum(1 for a in activity_log if a.get("status") == "error")

        c1, c2, c3 = st.columns(3)
        with c1: kpi_card("Total Tools Invoked", str(total_steps), "pipeline operations", "#380E16", "linear-gradient(90deg, #722F37, #A3485E)")
        with c2: kpi_card("Execution Latency", f"{total_time}ms", "end-to-end duration", "#722F37", "linear-gradient(90deg, #8B1E3F, #C4627A)")
        with c3: kpi_card("Pipeline Status", "Zero Errors" if errors == 0 else f"{errors} Failed", "100% reliability" if errors == 0 else "degraded", "#2D7A58" if errors == 0 else "#9C1D3A", "linear-gradient(90deg, #2D7A58, #4E9B78)")

        st.markdown("")

        # Activity timeline
        st.markdown('<div class="section-header">⚡ Agent Execution Sequence</div>', unsafe_allow_html=True)

        for i, activity in enumerate(activity_log):
            status = activity.get("status", "done")
            icon = "✅" if status == "done" else "❌" if status == "error" else "🔄"
            border_color = "#2D7A58" if status == "done" else "#9C1D3A" if status == "error" else "#BA6A24"
            duration = activity.get("duration_ms", 0)

            st.markdown(f"""
            <div style="display: flex; align-items: flex-start; margin: 6px 0;
                        background: linear-gradient(145deg, rgba(255, 255, 255, 0.95) 0%, rgba(252, 248, 244, 0.92) 100%);
                        border-left: 4px solid {border_color}; border: 1px solid rgba(114, 47, 55, 0.12);
                        border-left-width: 4px; border-radius: 0 12px 12px 0; padding: 14px 18px;
                        box-shadow: 0 6px 18px rgba(114, 47, 55, 0.05);">
                <div style="font-size: 18px; margin-right: 14px; min-width: 28px;">{icon}</div>
                <div style="flex: 1;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="font-family: 'Outfit', sans-serif; font-size: 15px; font-weight: 700; color: #380E16;">
                            Step {activity['step']}: {activity['tool']}
                        </div>
                        <div style="font-size: 12px; color: #8C6C72; font-family: 'JetBrains Mono', monospace;">
                            {activity.get('timestamp', '')} &nbsp;|&nbsp; <span style="color:#722F37; font-weight:700;">{duration}ms</span>
                        </div>
                    </div>
                    <div style="font-size: 13.5px; color: #4A1521; margin-top: 4px;">
                        {activity['description']}
                    </div>
                    {f'<div style="font-size: 12px; color: #8C6C72; margin-top: 4px; font-family: monospace;">Details: {activity["details"]}</div>'
                     if activity.get('details') else ''}
                </div>
            </div>
            """, unsafe_allow_html=True)
            if i < len(activity_log) - 1:
                st.markdown('<div style="text-align:left; padding-left:26px; color:rgba(114,47,55,0.4); font-size:16px;">↓</div>',
                            unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: REPORTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Reports":
    st.markdown('<div class="section-header">📊 Reports & Export Center</div>', unsafe_allow_html=True)

    if not st.session_state.analysis_done:
        st.info("📂 Upload a dataset to generate reports.")
    else:
        results = st.session_state.analysis_results
        df = results.get("df", pd.DataFrame())

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("""
            <div style="background: linear-gradient(145deg, rgba(255, 255, 255, 0.95) 0%, rgba(252, 248, 244, 0.92) 100%);
                        border: 1px solid rgba(114, 47, 55, 0.18);
                        border-radius: 16px; padding: 28px; text-align: center;
                        box-shadow: 0 12px 28px -8px rgba(114, 47, 55, 0.08);">
                <div style="font-size: 40px; margin-bottom: 12px;">📄</div>
                <div style="font-family: 'Outfit', sans-serif; font-size: 20px; font-weight: 700; color: #380E16; margin-bottom: 8px;">Executive PDF Report</div>
                <div style="font-size: 13.5px; color: #6E4D53; margin-bottom: 20px; line-height: 1.5;">
                    Synthesized executive brief including KPI distributions, detected root causes, severity matrices, and strategic recommendations.
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("")

            if st.button("📄 Generate & Download PDF Report", use_container_width=True):
                with st.spinner("Generating PDF..."):
                    pdf_bytes = report_generator.generate_pdf(
                        df=df,
                        quality_report=results.get("quality_report", {}),
                        recurring_issues=results.get("recurring_issues", []),
                        insights=results.get("insights", []),
                        recommendations=results.get("recommendations", []),
                        trend_data=results.get("trend_data", {}),
                    )
                if pdf_bytes:
                    st.download_button(
                        "⬇️ Download PDF",
                        data=pdf_bytes,
                        file_name="customer_feedback_analysis_report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                    st.success("✅ PDF report generated!")
                else:
                    st.warning("⚠️ PDF generation unavailable (ReportLab not installed). Install with: pip install reportlab")

        with c2:
            st.markdown("""
            <div style="background: linear-gradient(145deg, rgba(255, 255, 255, 0.95) 0%, rgba(252, 248, 244, 0.92) 100%);
                        border: 1px solid rgba(45, 122, 88, 0.25);
                        border-radius: 16px; padding: 28px; text-align: center;
                        box-shadow: 0 12px 28px -8px rgba(45, 122, 88, 0.08);">
                <div style="font-size: 40px; margin-bottom: 12px;">📊</div>
                <div style="font-family: 'Outfit', sans-serif; font-size: 20px; font-weight: 700; color: #380E16; margin-bottom: 8px;">Enriched Dataset (CSV)</div>
                <div style="font-size: 13.5px; color: #6E4D53; margin-bottom: 20px; line-height: 1.5;">
                    Download the processed dataset with attached sentiment polarities, emotions, topics, issue labels, and urgency scores.
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("")

            if st.button("📊 Download Analyzed CSV", use_container_width=True):
                csv_data = report_generator.generate_csv(df)
                st.download_button(
                    "⬇️ Download CSV",
                    data=csv_data,
                    file_name="analyzed_feedback.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
                st.success("✅ CSV ready for download!")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: FEEDBACK EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Feedback Explorer":
    st.markdown('<div class="section-header">🔎 Feedback Explorer</div>', unsafe_allow_html=True)
    st.markdown('<p style="color: #6E4D53;">Browse and filter all analyzed feedback records.</p>',
                unsafe_allow_html=True)

    if not st.session_state.analysis_done:
        st.info("📂 Upload a dataset to explore feedback records.")
    else:
        df = st.session_state.analysis_results.get("df", pd.DataFrame())
        feedback_col = st.session_state.feedback_col or "feedback"

        # Filters
        cf1, cf2, cf3, cf4 = st.columns(4)
        with cf1:
            sentiment_filter = st.multiselect(
                "Sentiment",
                options=["Positive", "Neutral", "Negative"] if "sentiment" in df.columns else [],
                default=["Positive", "Neutral", "Negative"] if "sentiment" in df.columns else []
            )
        with cf2:
            priority_filter = st.multiselect(
                "Priority",
                options=["HIGH", "MEDIUM", "LOW"] if "priority" in df.columns else [],
                default=["HIGH", "MEDIUM", "LOW"] if "priority" in df.columns else []
            )
        with cf3:
            topic_filter = st.multiselect(
                "Topic",
                options=sorted(df["topic"].unique().tolist()) if "topic" in df.columns else [],
                default=[]
            )
        with cf4:
            search_term = st.text_input("🔍 Search feedback text:", "")

        # Apply filters
        filtered_df = df.copy()
        if "sentiment" in df.columns and sentiment_filter:
            filtered_df = filtered_df[filtered_df["sentiment"].isin(sentiment_filter)]
        if "priority" in df.columns and priority_filter:
            filtered_df = filtered_df[filtered_df["priority"].isin(priority_filter)]
        if "topic" in df.columns and topic_filter:
            filtered_df = filtered_df[filtered_df["topic"].isin(topic_filter)]
        if search_term:
            filtered_df = filtered_df[
                filtered_df[feedback_col].astype(str).str.contains(search_term, case=False, na=False)
            ]

        st.markdown(f"**{len(filtered_df):,} records** matching current filters")

        # Select display columns
        display_cols = [feedback_col, "sentiment", "emotion", "topic",
                        "detected_issue", "priority", "sentiment_confidence"]
        available_cols = [c for c in display_cols if c in filtered_df.columns]

        # Style the dataframe
        if len(filtered_df) > 0:
            display_df = filtered_df[available_cols].head(200)
            st.dataframe(
                display_df,
                use_container_width=True,
                height=500,
            )
        else:
            st.warning("No records match the current filters.")


