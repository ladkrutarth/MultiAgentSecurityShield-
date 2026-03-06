"""
Veriscan — Unified AI Fraud & Compliance Dashboard
AI-Powered Fraud Detection | Dynamic Auth | CFPB Analysis & RAG
"""

import os
# ---------------------------------------------------------------------------
# System Stability Guards (Fixes SIGABRT on macOS Sequoia)
# ---------------------------------------------------------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import requests

# API Backend URL
API_BASE_URL = os.environ.get("VERISCAN_API_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

FEATURES_PATH = PROJECT_ROOT / "dataset" / "csv_data" / "financial_advisor_dataset.csv"
CFPB_PATH = PROJECT_ROOT / "dataset" / "csv_data" / "cfpb_credit_card.csv"

# ---------------------------------------------------------------------------
# Aesthetics & Accessibility Constants
# ---------------------------------------------------------------------------
CHART_TEXT_COLOR = "#0f172a"  # Darker Navy for maximum contrast
CHART_FONT = {"family": "Outfit", "size": 14, "color": CHART_TEXT_COLOR}

# Helper to apply unified accessible theme to Plotly figures
def apply_accessible_theme(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,250,252,0.5)",
        font=CHART_FONT,
        margin={"l": 30, "r": 30, "t": 60, "b": 30},
        title_font={"size": 20, "family": "Outfit", "color": CHART_TEXT_COLOR, "weight": "bold"},
        legend_font={"size": 13, "color": CHART_TEXT_COLOR},
    )
    fig.update_xaxes(
        showgrid=True, 
        gridcolor="rgba(226, 232, 240, 1)", 
        tickfont={"size": 12, "color": CHART_TEXT_COLOR, "weight": "bold"},
        title_font={"size": 14, "color": CHART_TEXT_COLOR, "weight": "bold"},
        linecolor="#94a3b8",
        linewidth=1
    )
    fig.update_yaxes(
        showgrid=True, 
        gridcolor="rgba(226, 232, 240, 1)", 
        tickfont={"size": 12, "color": CHART_TEXT_COLOR, "weight": "bold"},
        title_font={"size": 14, "color": CHART_TEXT_COLOR, "weight": "bold"},
        linecolor="#94a3b8",
        linewidth=1
    )
    return fig

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Veriscan — Unified Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS (Soft Light-Glass Aesthetic)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

    /* Soft Light-Glass Aesthetic (Port 8507 Restoration) */
    .stApp {
        background: radial-gradient(at 0% 0%, rgba(102, 126, 234, 0.1) 0, transparent 50%), 
                    radial-gradient(at 100% 0%, rgba(118, 75, 162, 0.1) 0, transparent 50%), 
                    #f8fafc;
        font-family: 'Outfit', sans-serif;
        color: #1e293b;
    }

    /* Frosted Glass Header */
    .main-header {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.5);
        padding: 2.5rem 3.5rem;
        border-radius: 24px;
        margin-bottom: 2.5rem;
        box-shadow: 0 10px 40px -10px rgba(0, 0, 0, 0.1);
    }

    .main-header h1 {
        margin: 0;
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1.5px;
    }

    /* Frosted Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.82);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.6);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        box-shadow: 0 4px 20px -5px rgba(0,0,0,0.05);
    }

    .metric-card:hover { 
        transform: translateY(-5px); 
        box-shadow: 0 15px 35px -10px rgba(0,0,0,0.1);
        border-color: rgba(99, 102, 241, 0.4);
    }
    .metric-card h3 { margin: 0; font-size: 0.9rem; text-transform: uppercase; color: #64748b; font-weight: 700; }
    .metric-card .value { font-size: 2.6rem; font-weight: 800; margin: 0.5rem 0; color: #0f172a; }

    /* Frosted Risk Badges */
    .risk-badge { display: inline-block; padding: 0.5rem 1.2rem; border-radius: 50px; font-size: 0.8rem; font-weight: 700; text-transform: uppercase; }
    .risk-low { background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }
    .risk-medium { background: #fef9c3; color: #854d0e; border: 1px solid #fef08a; }
    .risk-high { background: #ffedd5; color: #9a3412; border: 1px solid #fed7aa; }
    .risk-critical { background: #fee2e2; color: #991b1b; border: 1px solid #fecaca; animation: pulse-light 2s infinite; }
    
    @keyframes pulse-light { 0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); } 70% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); } 100% { box-shadow:0 0 0 0 rgba(239, 68, 68, 0); } }

    /* Frosted Feature Cards */
    .feature-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.8);
        border-radius: 18px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 30px rgba(0,0,0,0.04);
    }

    /* High-Contrast Labeling */
    .stMarkdown h4 {
        color: #0f172a !important;
        font-weight: 700 !important;
    }

    /* Softer High-Contrast Labels */
    div[data-testid="stTextInput"] label p {
        color: #1e293b !important;
        font-weight: 500 !important; /* Normal weight as requested */
        font-size: 1.1rem !important;
    }

    /* Premium Soft-Indigo Input Box - Strong Override */
    div[data-testid="stTextInput"] div[data-baseweb="input"],
    div[data-testid="stTextArea"] div[data-baseweb="textarea"],
    div[data-baseweb="input"] {
        background-color: #ffffff !important;
        border: 2px solid #818cf8 !important;
        border-radius: 10px !important;
    }

    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea,
    div[data-baseweb="input"] input {
        color: #0f172a !important; /* High Contrast Navy */
        -webkit-text-fill-color: #0f172a !important;
        background-color: #ffffff !important;
        font-weight: 600 !important;
        caret-color: #0f172a !important;
    }

    div[data-baseweb="input"]:focus-within {
        border-color: #6366f1 !important;
        box-shadow: 0 0 10px rgba(99, 102, 241, 0.2) !important;
        background-color: #ffffff !important;
    }

    /* Premium Button Scaling */
    div.stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 700 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
    }
    div.stButton > button:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4) !important;
    }

    .rag-answer {
        background: #ffffff;
        border: 2px solid #6366f1;
        padding: 2rem;
        border-radius: 12px;
        font-size: 1.1rem;
        line-height: 1.8;
        color: #0f172a;  /* Extreme High Contrast Navy */
        margin-top: 2rem;
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.1);
    }

    /* Tabs Override */
    div[data-testid="stTabs"] button {
        color: #64748b;
        font-weight: 600;
        font-size: 1.1rem;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #4f46e5 !important;
    }

    /* Sidebar High-Contrast Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li,
    section[data-testid="stSidebar"] label {
        color: #1e293b !important;
        font-weight: 500 !important;
    }
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: #0f172a !important;
        font-weight: 700 !important;
    }

    /* High-Contrast Widget Labels & Info Text */
    div[data-testid="stWidgetLabel"] p,
    .stAlert p,
    table, th, td, 
    div[data-testid="stTable"] td, 
    div[data-testid="stTable"] th {
        color: #000000 !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
    }

    /* High-Contrast Radio/Multiple Choice options */
    div[data-testid="stRadio"] label p,
    div[data-testid="stRadio"] div[role="radiogroup"] label p {
        color: #000000 !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
    }
</style>
""", unsafe_allow_html=True)

import random

# ---------------------------------------------------------------------------
# Data & Resource Loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600)
def load_fraud_data(path: Path, mtime: float):
    if path.exists():
        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]
        # Compatibility fix: ensure is_fraud_flag exists if it's the advisor dataset
        if 'is_fraud_flag' not in df.columns:
            if 'is_fraud' in df.columns:
                df = df.rename(columns={'is_fraud': 'is_fraud_flag'})
            elif 'is_fraud_flag' not in df.columns:
                # Last resort: create the column if missing
                df['is_fraud_flag'] = False
        return df
    return pd.DataFrame()

@st.cache_data(ttl=600)
def load_cfpb_data():
    if CFPB_PATH.exists():
        # Large scale analysis enabled
        return pd.read_csv(CFPB_PATH, nrows=50000)
    return pd.DataFrame()

def api_available() -> bool:
    """Check if the FastAPI backend is running."""
    try:
        r = requests.get(f"{API_BASE_URL}/api/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False

@st.cache_resource
def get_llm_via_api(prompt: str, max_tokens: int = 500) -> str:
    """Calls the backend API for LLM generation to avoid GPU contention in the frontend."""
    try:
        resp = requests.post(
            f"{API_BASE_URL}/api/llm/generate",
            json={"prompt": prompt, "max_tokens": max_tokens},
            timeout=60
        )
        if resp.status_code == 200:
            return resp.json().get("response", "Error: No response from API.")
        return f"Error: API returned {resp.status_code}"
    except Exception as e:
        return f"Error connecting to AI Backend: {e}"

# ---------------------------------------------------------------------------
# Aesthetics & Accessibility Helpers
# ---------------------------------------------------------------------------

def risk_badge_html(level: str) -> str:
    cls = f"risk-{level.lower()}"
    return f'<span class="risk-badge {cls}">{level}</span>'

# ---------------------------------------------------------------------------
# Sidebar UI
# ---------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown("<h2 style='color:#0f172a;'>🛡️ Veriscan</h2>", unsafe_allow_html=True)
        st.markdown("Security & Intelligence Hub")
        st.divider()

        st.divider()
        st.markdown("### System Status")
        fraud_df = load_fraud_data(FEATURES_PATH, FEATURES_PATH.stat().st_mtime if FEATURES_PATH.exists() else 0)
        cfpb_df = load_cfpb_data()
        
        st.markdown(f"""
        - 💸 Transactions: **{len(fraud_df):,}**
        - 📝 Complaints: **{len(cfpb_df):,}**
        - 🤖 Intelligence: **Local MLX-LM**
        """)

# ---------------------------------------------------------------------------
# Tab 1: Fraud Dashboard
# ---------------------------------------------------------------------------
def render_dashboard_tab(df):
    if df.empty:
        st.warning("Fraud dataset not found.")
        return

    st.markdown("### 📊 Market Fraud Overview")
    
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(f"<div class='metric-card'><h3>Processed</h3><div class='value'>{len(df):,}</div></div>", unsafe_allow_html=True)
    with c2: st.markdown(f"<div class='metric-card'><h3>Fraud Cases</h3><div class='value' style='color:#dc2626'>{df['is_fraud_flag'].sum():,}</div></div>", unsafe_allow_html=True)
    with c3: st.markdown(f"<div class='metric-card'><h3>Fraud Risk</h3><div class='value'>{(df['is_fraud_flag'].mean()*100):.1f}%</div></div>", unsafe_allow_html=True)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Fraud Heatmap by Category")
        cat_data = df.groupby('category')['is_fraud_flag'].sum().sort_values(ascending=False).reset_index()
        fig = px.bar(cat_data, x='is_fraud_flag', y='category', orientation='h', color='is_fraud_flag', 
                     color_continuous_scale='Blues', template='plotly_white')
        st.plotly_chart(apply_accessible_theme(fig), use_container_width=True)
    
    with col2:
        st.subheader("Interactive Fraud Map (US)")
        state_data = df.groupby('state')['is_fraud_flag'].sum().reset_index()
        fig = px.choropleth(
            state_data,
            locations='state',
            locationmode="USA-states",
            color='is_fraud_flag',
            color_continuous_scale="OrRd",
            scope="usa",
            labels={'is_fraud_flag': 'Fraud Cases'}
        )
        st.plotly_chart(apply_accessible_theme(fig), use_container_width=True)

    st.subheader("Monthly Fraud Trends (Month & Year)")
    trend_data = df[df['is_fraud_flag'] == True].groupby('month_key').size().reset_index(name='fraud_count')
    fig_trend = px.line(trend_data, x='month_key', y='fraud_count', 
                        title="Fraud Cases Trend (2-Year History)",
                        labels={'month_key': 'Month', 'fraud_count': 'Number of Fraud Cases'},
                        markers=True, template='plotly_white')
    fig_trend.update_traces(line_color='#dc2626', line_width=3)
    st.plotly_chart(apply_accessible_theme(fig_trend), use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 3: CFPB Market Intelligence
# ---------------------------------------------------------------------------
def render_cfpb_tab(df):
    st.markdown("### 🔍 CFPB Credit Card Intelligence")
    if df.empty:
        st.warning("CFPB dataset missing or empty.")
        return

    reporting_cos = [
        "EQUIFAX, INC.", 
        "TRANSUNION INTERMEDIATE HOLDINGS, INC.", 
        "EXPERIAN INFORMATION SOLUTIONS, INC.",
        "LEXISNEXIS RISK DATA MANAGEMENT INC."
    ]
    
    m1, m2 = st.columns([1, 1])
    with m1:
        st.markdown("#### 🏛️ Reporting Agencies")
        rep_df = df[df['Company'].isin(reporting_cos)]
        rep_counts = rep_df['Company'].value_counts().reset_index()
        fig_rep = px.bar(rep_counts, x='count', y='Company', orientation='h', template='plotly_white', 
                         color='count', color_continuous_scale='Turbo', title="Credit Reporting Volume")
        st.plotly_chart(apply_accessible_theme(fig_rep), use_container_width=True)

    with m2:
        st.markdown("#### 💳 Card Issuers")
        iss_df = df[~df['Company'].isin(reporting_cos)]
        iss_counts = iss_df['Company'].value_counts().head(8).reset_index()
        fig_iss = px.bar(iss_counts, x='count', y='Company', orientation='h', template='plotly_white', 
                         color='count', color_continuous_scale='IceFire', title="Top Issuer Complaints")
        st.plotly_chart(apply_accessible_theme(fig_iss), use_container_width=True)

    st.divider()
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("#### 🗺️ Geographic Distribution")
        state_data = df.groupby('State').size().reset_index(name='Complaints')
        fig2 = px.choropleth(
            state_data,
            locations='State',
            locationmode="USA-states",
            color='Complaints',
            color_continuous_scale="Viridis",
            scope="usa",
            labels={'Complaints': 'Complaints'},
            title="Complaint Distribution by State"
        )
        st.plotly_chart(apply_accessible_theme(fig2), use_container_width=True)

    with c2:
        st.markdown("#### 💬 AI Intelligence & Search")
        st.info("Query the knowledge base for fraud patterns or policy details.")
        query = st.text_input("Search context:", placeholder="e.g. Find high-value travel anomalies...", key="cfpb_rag_q")
        if st.button("Query Knowledge Base", key="cfpb_rag_btn") and query:
            with st.spinner("Analyzing..."):
                try:
                    resp = requests.post(f"{API_BASE_URL}/api/rag/query", json={"query": query, "n_results": 5}, timeout=30)
                    if resp.status_code == 200:
                        data = resp.json()
                        if data["results"]:
                            for r in data["results"]:
                                st.markdown(f"<div class='rag-answer'>{r}</div>", unsafe_allow_html=True)
                        else:
                            st.info("No matching context found.")
                except Exception as e:
                    st.error(f"RAG Error: {e}")



# ---------------------------------------------------------------------------
# Tab 1: AI Specialist Models (Security & Financial)
# ---------------------------------------------------------------------------
def render_omni_tab():
    from agents.financial_advisor_agent import FinancialAdvisorAgent
    
    st.markdown("### 🤖 AI Specialist Models")
    st.caption("Select your specialized AI agent for focused analysis.")

    # 1. User Context Selector
    try:
        adv = FinancialAdvisorAgent()
        all_users = adv.get_all_users()
    except Exception:
        all_users = ["USER_0001"]

    colX, colY = st.columns([1, 2])
    with colX:
        st.markdown("<p style='color: black; font-weight: bold; font-size: 14px; margin-bottom: 5px; margin-left: 2px;'>👤 Target Context:</p>", unsafe_allow_html=True)
        selected_user = st.selectbox("", all_users[:50], key="omni_user", label_visibility="collapsed")
    
    with colY:
        st.markdown("<p style='color: black; font-weight: bold; font-size: 14px; margin-bottom: 5px; margin-left: 2px;'>🧠 Select AI Model:</p>", unsafe_allow_html=True)
        selected_model = st.selectbox(
            "",
            ["🛡️ Security AI Analyst", "💰 Financial AI Advisor"],
            key="model_selector",
            label_visibility="collapsed"
        )
        
    is_security_model = "Security" in selected_model
    is_financial_model = "Financial" in selected_model

    # 2. Dynamic Top Widget (Shield vs. Chart preview)
    if is_security_model:
        try:
            monitor = adv.tool_suspicious_activity_monitor(selected_user)
            if monitor["alert_count"] > 0:
                overall = monitor["overall_status"]
                border = "#dc2626" if "CRITICAL" in overall else "#d97706"
                st.markdown(
                    f"""<div style='background:rgba(220,38,38,0.05);border:1px solid {border};border-radius:12px;padding:1rem;margin-bottom:1.5rem;'>
                    <div style='font-weight:700;font-size:1rem;margin-bottom:0.5rem;'>🛡️ System Shield Live Monitor — <span style='color:{border};'>{overall}</span></div>
                    <p style='font-size:0.9rem; margin-bottom:0.5rem;'>Detected {monitor['alert_count']} anomalies requiring immediate attention.</p>
                    </div>""",
                    unsafe_allow_html=True,
                )
            else:
                st.success("✅ **System Shield:** Active monitoring shows no suspicious activity.", icon="🛡️")
        except Exception:
            pass
    elif is_financial_model:
        try:
            summary = adv.tool_spending_summary(selected_user)
            st.info(f"📊 **Quick Review:** {summary['archetype'].replace('_', ' ').title()} spender. Monthly Avg: **${summary['avg_monthly_spend']:,.2f}**")
        except Exception:
            pass

    # 3. Dedicated Chat & Investigation Interface
    st.markdown(f"#### 💬 {selected_model} Interface")
    
    if "run_omni" not in st.session_state:
        st.session_state.run_omni = False

    def set_preset(query):
        st.session_state.omni_input = query
        st.session_state.run_omni = True

    # Dynamic Quick Answer Buttons based on model
    if is_security_model:
        qa1, qa2 = st.columns(2)
        qa1.button("🔍 Run Fraud Scan", on_click=set_preset, args=("Scan my recent transactions for any signs of fraud.",), use_container_width=True, key="sec_qa1")
        qa2.button("🛡️ Anomaly Protocol", on_click=set_preset, args=("Execute the anomaly detection protocol and report risks.",), use_container_width=True, key="sec_qa2")
        placeholder = "e.g. 'Investigate the recent high-value transactions for fraud alerts.'"
    else:
        qa1, qa2, qa3 = st.columns(3)
        qa1.button("📊 Spend Review", on_click=set_preset, args=("Provide a full spending portfolio review.",), use_container_width=True, key="fin_qa1")
        qa2.button("💰 Savings Plan", on_click=set_preset, args=("Generate a targeted savings strategy for my archetype.",), use_container_width=True, key="fin_qa2")
        qa3.button("💳 Credit Impact", on_click=set_preset, args=("Estimate my credit health and identifies risk factors.",), use_container_width=True, key="fin_qa3")
        placeholder = "e.g. 'How much did I spend on coffee this month, and how can I save?'"

    user_q = st.text_area("Request Intelligence:", 
                         placeholder=placeholder,
                         height=100, key="omni_input")
    
    execute_clicked = st.button(f"🚀 Execute {selected_model.split()[1]} Reasoning", key="omni_btn", type="primary")
    
    if (execute_clicked or st.session_state.run_omni) and user_q:
        st.session_state.run_omni = False  # Reset the trigger
        with st.spinner(f"{selected_model.split()[1]} Agent is analyzing..."):
            try:
                if is_security_model:
                    # Route to dedicated Security Endpoint
                    resp = requests.post(f"{API_BASE_URL}/api/security/chat", json={"message": user_q}, timeout=120)
                else:
                    # Route strictly to Financial Advisor
                    resp = requests.post(f"{API_BASE_URL}/api/advisor/chat", json={"user_id": selected_user, "message": user_q}, timeout=120)

                if resp.status_code == 200:
                    res = resp.json()
                    
                    st.info(f"🧠 Reasoning Mode: **{selected_model}**")
                    
                    # Show final response with proper markdown rendering
                    with st.container():
                        st.markdown(res["reply"])

                    # Show Chart *only* for Financial Advisor if requested
                    if is_financial_model and res.get("show_chart") and res.get("chart_data"):
                        with st.container():
                            st.markdown("#### 📊 Diagnostic Portfolio Visualization")
                            data = res["chart_data"]
                            fig = px.bar(
                                x=list(data.keys()), 
                                y=list(data.values()), 
                                labels={'x': 'Category', 'y': 'Total Allocation ($)'},
                                color=list(data.values()),
                                color_continuous_scale="Sunset",
                                text_auto='.2s'
                            )
                            fig.update_layout(
                                height=400, 
                                margin=dict(l=20, r=20, t=20, b=20),
                                paper_bgcolor="white",
                                plot_bgcolor="white",
                                font=dict(family="Outfit, sans-serif", size=13, color="#000000"),
                                coloraxis_showscale=False,
                                title_font=dict(color="#000000")
                            )
                            fig.update_xaxes(title_font=dict(color="#000000"), tickfont=dict(color="#000000"))
                            fig.update_yaxes(title_font=dict(color="#000000"), tickfont=dict(color="#000000"))
                            fig.update_traces(textposition='outside', cliponaxis=False, textfont=dict(color="#000000"))
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Trace rendering removed to enforce a clean, professional, sub-300 word summary via the Agent.
                else:
                    st.error(f"API Error: {resp.status_code}")
            except Exception as e:
                st.error(f"🔴 Connection Error: {e}")

    # =======================================================================
    # Bottom Domain Diagnostics (Context Aware)
    # =======================================================================
    st.divider()
    st.markdown("#### 💡 Diagnostic Shortcuts")
    
    if is_financial_model:
        st.info("Visualizing deep financial diagnostics.")
        # ── Vector Analysis: Instant local chart ──────────────────────────────
        if st.button("📉 Compute Vector Analysis", use_container_width=True):
            with st.spinner("Computing spending vectors..."):
                try:
                    adv_local = FinancialAdvisorAgent()
                    chart_data = adv_local.get_chart_data(selected_user)
                    summary = adv_local.tool_spending_summary(selected_user)
    
                    if chart_data:
                        st.markdown("#### 📊 Spending Vector Analysis")
                        fig = px.bar(
                            x=list(chart_data.keys()),
                            y=list(chart_data.values()),
                            labels={'x': 'Category', 'y': 'Total Spend ($)'},
                            color=list(chart_data.values()),
                            color_continuous_scale="Sunset",
                            text_auto='.2s'
                        )
                        fig.update_layout(
                            height=400,
                            margin=dict(l=20, r=20, t=30, b=20),
                            paper_bgcolor="white",
                            plot_bgcolor="white",
                            font=dict(family="Outfit, sans-serif", size=13, color="#000000"),
                            coloraxis_showscale=False
                        )
                        fig.update_xaxes(title_font=dict(color="#000000"), tickfont=dict(color="#000000"))
                        fig.update_yaxes(title_font=dict(color="#000000"), tickfont=dict(color="#000000"))
                        fig.update_traces(textposition='outside', cliponaxis=False, textfont=dict(color="#000000"))
                        st.plotly_chart(fig, use_container_width=True)
    
                        # Quick summary metrics - Force Black Text
                        c1, c2, c3 = st.columns(3)
                        c1.markdown(f"<div style='color: black; font-size: 0.9em; opacity: 0.8;'>💰 Total Spend</div><div style='color: black; font-size: 1.8em; font-weight: bold;'>${summary.get('total_spend', 0):,.2f}</div>", unsafe_allow_html=True)
                        c2.markdown(f"<div style='color: black; font-size: 0.9em; opacity: 0.8;'>📅 Monthly Avg</div><div style='color: black; font-size: 1.8em; font-weight: bold;'>${summary.get('avg_monthly_spend', 0):,.2f}</div>", unsafe_allow_html=True)
                        c3.markdown(f"<div style='color: black; font-size: 0.9em; opacity: 0.8;'>🏷️ Top Merchant</div><div style='color: black; font-size: 1.8em; font-weight: bold;'>{summary.get('top_merchant', 'N/A')}</div>", unsafe_allow_html=True)
                    else:
                        st.warning("No spending data found for this user.")
                except Exception as e:
                    st.error(f"Vector Analysis Error: {e}")
    else:
        st.info("Running localized security sweeps.")
        # ── Anomaly Protocol: Instant local fraud scan ────────────────────────
        if st.button("🔍 Execute Anomaly Protocol", use_container_width=True):
            with st.spinner("Scanning for anomalies..."):
                try:
                    from agents.financial_advisor_agent import FinancialAdvisorAgent
                    adv_local = FinancialAdvisorAgent()
                    fraud = adv_local.tool_realtime_fraud_check(selected_user)
                    monitor = adv_local.tool_suspicious_activity_monitor(selected_user)

                    status_fraud = fraud.get("overall_status", "✅ CLEAR")
                    status_monitor = monitor.get("overall_status", "✅ CLEAR")
                    alert_count = fraud.get("alerts_found", 0) + monitor.get("alert_count", 0)
                    avg_risk = fraud.get("avg_risk_score", 0)

                    # Build concise ~150-word report from real data
                    report_parts = [
                        f"**Anomaly Detection Protocol — Complete**\n",
                        f"**Fraud Scan:** {status_fraud}",
                        f"**Activity Monitor:** {status_monitor}",
                        f"**Transactions Scanned:** {fraud.get('transactions_scanned', 0)} | **Alerts Found:** {alert_count} | **Avg Risk Score:** {avg_risk:.3f}\n",
                    ]

                    if fraud.get("alerts"):
                        report_parts.append("**⚠️ Flagged Transactions:**")
                        for a in fraud["alerts"][:5]:
                            flags_str = ", ".join(a["flags"][:2])
                            report_parts.append(f"- {a['merchant']} (${a['amount']}): {flags_str}")
                    elif monitor.get("alerts"):
                        report_parts.append("**⚠️ Suspicious Activity:**")
                        for a in monitor["alerts"][:5]:
                            report_parts.append(f"- {a['title']}: {a['detail']}")

                    st.markdown("\n".join(report_parts))
                except Exception as e:
                    st.error(f"Anomaly Protocol Error: {e}")






# ---------------------------------------------------------------------------
# Tab 6: Spending DNA — Financial Fingerprint
# ---------------------------------------------------------------------------
def render_dna_tab():
    st.markdown("### 🧬 Spending DNA — Financial Fingerprint")
    st.info("Every user has a unique 8-axis financial signature. This is used for identity verification and anomaly detection.")

    try:
        from agents.spending_dna_agent import SpendingDNAAgent
        dna_agent = SpendingDNAAgent()
        all_users = dna_agent.get_all_users()
    except Exception as e:
        st.error(f"Could not load DNA agent: {e}")
        return

    col_sel, col_trust = st.columns([2, 1])
    with col_sel:
        selected = st.selectbox("Select User for DNA Profile:", all_users[:30], key="dna_user")

    dna = dna_agent.compute_dna(selected)
    if "error" in dna:
        st.error(dna["error"])
        return

    with col_trust:
        score = dna["avg_trust_score"]
        color = "#16a34a" if score >= 0.8 else ("#d97706" if score >= 0.6 else "#dc2626")
        st.markdown(f"""
        <div class='metric-card' style='margin-top:1.5rem;'>
            <h3>Trust Score</h3>
            <div class='value' style='color:{color};'>{score:.0%}</div>
            <p style='font-size:0.8rem;color:#64748b;'>{dna['trust_grade']}</p>
        </div>
        """, unsafe_allow_html=True)

    col_radar, col_stats = st.columns([3, 2])
    with col_radar:
        # Radar chart
        labels = dna["radar_labels"]
        values = dna["radar_values"]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=labels + [labels[0]],
            fill="toself",
            fillcolor="rgba(79, 70, 229, 0.3)",
            line=dict(color="#4338ca", width=4),
            name=selected,
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=12, color="#0f172a", weight="bold")),
                angularaxis=dict(tickfont=dict(size=13, color="#0f172a", weight="bold")),
                bgcolor="rgba(248,250,252,1)",
            ),
            showlegend=False,
            title=dict(text=f"🧬 {selected} — Spending DNA", font=dict(size=18, color="#0f172a", weight="bold")),
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=70, r=70, t=70, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_stats:
        st.markdown("#### DNA Axes (Raw Values)")
        raw = dna["raw_axes"]
        axis_df = pd.DataFrame([
            {"Axis": label, "Raw Value": round(raw.get(col, 0), 3), "Normalized": round(val, 3)}
            for (col, label), val in zip(
                [("avg_txn_amount","Avg Txn Amount"),("location_entropy","Location Entropy"),
                 ("weekend_ratio","Weekend Ratio"),("category_diversity","Category Diversity"),
                 ("time_of_day_pref","Time of Day Pref"),("risk_appetite_score","Risk Appetite"),
                 ("spending_velocity","Spending Velocity"),("merchant_loyalty_score","Merchant Loyalty")],
                dna["radar_values"]
            )
        ])
        st.dataframe(axis_df, use_container_width=True, hide_index=True)

        st.markdown(f"**Time Preference:** {dna['time_preference']}")
        st.markdown(f"**Anomalous Sessions:** {dna['anomalous_count']:,} / {dna['total_sessions']:,}")

    # Session comparison
    st.divider()
    st.markdown("#### 🔍 Session vs. DNA Comparison")
    if st.button("Simulate New Session", key="dna_compare"):
        comparison = dna_agent.compare_session(selected)
        verdict_color = "#16a34a" if "Trusted" in comparison["verdict"] else ("#d97706" if "Moderate" in comparison["verdict"] else "#dc2626")

        st.markdown(f"<div class='rag-answer' style='border-color:{verdict_color};'><strong>{comparison['verdict']}</strong><br>Session Trust Score: <strong>{comparison['session_trust_score']:.0%}</strong> | Composite Deviation: {comparison['composite_deviation']:.3f}</div>", unsafe_allow_html=True)

        # Overlay radar
        fig2 = go.Figure()
        fig2.add_trace(go.Scatterpolar(
            r=comparison["baseline_radar"] + [comparison["baseline_radar"][0]],
            theta=comparison["radar_labels"] + [comparison["radar_labels"][0]],
            fill="toself", fillcolor="rgba(99,102,241,0.2)", line=dict(color="#6366f1", width=2, dash="dash"), name="DNA Baseline",
        ))
        fig2.add_trace(go.Scatterpolar(
            r=comparison["session_radar"] + [comparison["session_radar"][0]],
            theta=comparison["radar_labels"] + [comparison["radar_labels"][0]],
            fill="toself", fillcolor="rgba(220,38,38,0.15)", line=dict(color="#dc2626", width=3), name="Current Session",
        ))
        fig2.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1]), bgcolor="rgba(248,250,252,0.8)"),
            showlegend=True, title="Session vs. DNA Baseline",
            paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=60, r=60, t=50, b=30),
        )
        st.plotly_chart(fig2, use_container_width=True)



# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def main():
    render_sidebar()

    st.markdown("""
    <div class="main-header">
        <h1>Veriscan Dashboard</h1>
        <p>Real-time Fraud Prevention &amp; Consumer Compliance Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs([
        "🤖 AI Specialists",
        "📊 Market Dash",
        "🔍 CFPB Market Intel",
        "🧬 Spending DNA",
    ])

    fraud_df = load_fraud_data(FEATURES_PATH, FEATURES_PATH.stat().st_mtime if FEATURES_PATH.exists() else 0)
    cfpb_df  = load_cfpb_data()

    with tabs[0]: render_omni_tab()
    with tabs[1]: render_dashboard_tab(fraud_df)
    with tabs[2]: render_cfpb_tab(cfpb_df)
    with tabs[3]: render_dna_tab()

if __name__ == "__main__":
    main()
