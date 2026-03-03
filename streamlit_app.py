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
import re
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

FEATURES_PATH = PROJECT_ROOT / "dataset" / "csv_data" / "processed_fraud_train.csv"
CFPB_PATH = PROJECT_ROOT / "dataset" / "csv_data" / "cfpb_credit_card.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_model_rf.joblib"
ENCODERS_PATH = PROJECT_ROOT / "models" / "encoders.joblib"

# ---------------------------------------------------------------------------
# Aesthetics & Accessibility Constants
# ---------------------------------------------------------------------------
CHART_TEXT_COLOR = "#1e293b"  # High contrast Navy for legibility
CHART_FONT = {"family": "Outfit", "size": 13, "color": CHART_TEXT_COLOR}

# Helper to apply unified accessible theme to Plotly figures
def apply_accessible_theme(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=CHART_FONT,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        title_font={"size": 18, "family": "Outfit", "color": CHART_TEXT_COLOR},
        legend_font={"size": 12, "color": CHART_TEXT_COLOR},
    )
    fig.update_xaxes(
        showgrid=True, 
        gridcolor="rgba(0,0,0,0.05)", 
        tickfont={"size": 11, "color": CHART_TEXT_COLOR},
        title_font={"size": 13, "color": CHART_TEXT_COLOR}
    )
    fig.update_yaxes(
        showgrid=True, 
        gridcolor="rgba(0,0,0,0.05)", 
        tickfont={"size": 11, "color": CHART_TEXT_COLOR},
        title_font={"size": 13, "color": CHART_TEXT_COLOR}
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
@st.cache_resource
def load_fraud_model():
    if MODEL_PATH.exists() and ENCODERS_PATH.exists():
        return joblib.load(MODEL_PATH), joblib.load(ENCODERS_PATH)
    return None, None

@st.cache_data(ttl=600)
def load_fraud_data():
    return pd.read_csv(FEATURES_PATH) if FEATURES_PATH.exists() else pd.DataFrame()

@st.cache_data(ttl=600)
def load_cfpb_data():
    if CFPB_PATH.exists():
        # Using subset for quick performance
        return pd.read_csv(CFPB_PATH, nrows=10000)
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
        fraud_df = load_fraud_data()
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
    with c2: st.markdown(f"<div class='metric-card'><h3>Fraud Cases</h3><div class='value' style='color:#dc2626'>{df['is_fraud'].sum():,}</div></div>", unsafe_allow_html=True)
    with c3: st.markdown(f"<div class='metric-card'><h3>Fraud Risk</h3><div class='value'>{(df['is_fraud'].mean()*100):.1f}%</div></div>", unsafe_allow_html=True)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Fraud by Category")
        cat_data = df.groupby('category')['is_fraud'].sum().sort_values(ascending=False).reset_index()
        fig = px.bar(cat_data, x='is_fraud', y='category', orientation='h', color='is_fraud', 
                     color_continuous_scale='Viridis', template='plotly_white', title="Complaint Volume by Category")
        st.plotly_chart(apply_accessible_theme(fig), use_container_width=True)
    
    with col2:
        st.subheader("Top Fraud States")
        state_data = df.groupby('state')['is_fraud'].sum().sort_values(ascending=False).head(10).reset_index()
        fig = px.bar(state_data, x='is_fraud', y='state', orientation='h', color='is_fraud', 
                     color_continuous_scale='Plasma', template='plotly_white', title="Top 10 High-Risk States")
        st.plotly_chart(apply_accessible_theme(fig), use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 2: AI Fraud Prediction
# ---------------------------------------------------------------------------
def render_prediction_tab(df, model, encoders):
    if not model:
        st.error("ML Model files missing. Please run training script first.")
        return

    st.markdown("### 🚨 AI Risk Assessment")
    
    # High-Contrast Selection Controls
    c1, c2 = st.columns([1, 2])
    with c1:
        risk_filter = st.selectbox("Target Risk Level:", ["Show All", "LOW", "MEDIUM", "HIGH", "CRITICAL"])
    
    # Efficient Filtering via Batch Inference
    if risk_filter != "Show All":
        with st.spinner(f"Scanning sample for {risk_filter} risk cases..."):
            subset = df.head(5000) # Use top 5000 for consistent responsiveness
            features = ["category", "amt", "gender", "state", "merchant", "hour", "day_of_week"]
            X = subset[features].copy()
            for c in ["category", "gender", "state", "merchant"]:
                X[c] = encoders[c].transform(X[c].astype(str))
            
            probs = model.predict_proba(X)[:, 1]
            subset['predicted_risk'] = ["CRITICAL" if p > 0.8 else "HIGH" if p > 0.5 else "MEDIUM" if p > 0.2 else "LOW" for p in probs]
            display_df = subset[subset['predicted_risk'] == risk_filter]
            
            if display_df.empty:
                st.warning(f"No {risk_filter} risk transactions found in the scanned batch. Try another level.")
                return
    else:
        display_df = df

    total_found = len(display_df)
    st.info(f"Showing {total_found:,} transactions matching criteria.")
    
    idx = st.slider("Select Transaction from Pool", 0, max(0, total_found-1), 0)
    row = display_df.iloc[idx]

    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
        st.markdown("#### Transaction Details")
        st.write(f"🏷️ **Merchant**: {row['merchant']}")
        st.write(f"📁 **Category**: {row['category']}")
        st.write(f"💰 **Amount**: ${row['amt']:.2f}")
        st.write(f"👤 **Holder**: {row['first']} {row['last']}")
        st.write(f"📍 **Location**: {row['state']}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        features = ["category", "amt", "gender", "state", "merchant", "hour", "day_of_week"]
        input_row = row[features].copy()
        for c in ["category", "gender", "state", "merchant"]:
            input_row[c] = encoders[c].transform([str(input_row[c])])[0]
        
        prob = model.predict_proba(pd.DataFrame([input_row]))[0][1]
        risk = "CRITICAL" if prob > 0.8 else "HIGH" if prob > 0.5 else "MEDIUM" if prob > 0.2 else "LOW"

        st.markdown(f"""
        <div class='metric-card' style='padding: 3rem;'>
            <h3 style='font-size: 1.1rem;'>ML Predicted Risk</h3>
            <div class='value' style='font-size: 5rem; color:{"#dc2626" if prob > 0.5 else "#16a34a"};'>{prob:.1%}</div>
            <div style='margin-top: 1rem;'>{risk_badge_html(risk)}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Model Insights")
    feat_df = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=True)
    fig = px.bar(feat_df, x='Importance', y='Feature', orientation='h', template="plotly_white", 
                 color='Importance', color_continuous_scale='Turbo', title="ML Feature Contribution Analysis")
    st.plotly_chart(apply_accessible_theme(fig), use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 3: Dynamic Authentication
# ---------------------------------------------------------------------------
def render_auth_tab(df):
    st.markdown("### 🔑 Context-Aware identity Auth")
    st.info("The system generates 3 dynamic questions based on recent account activity. You must answer correctly to verify your identity.")

    # User pool for demo
    user_counts = df.groupby(['first', 'last']).size().reset_index(name='c').query('c > 4').head(12)
    user_list = [f"{r['first']} {r['last']}" for _, r in user_counts.iterrows()]
    selected_user = st.selectbox("Simulate Authentication for User:", user_list)
    first_name, last_name = selected_user.split(" ", 1)

    # State management for the 3-question quiz
    if "auth_quiz" not in st.session_state or st.session_state.get("auth_user") != selected_user:
        st.session_state.auth_quiz = {
            "step": 0,
            "questions": [],
            "score": 0,
            "completed": False,
            "last_result": None,
            "user_responses": []
        }
        st.session_state.auth_user = selected_user

    quiz = st.session_state.auth_quiz

    if not quiz["completed"]:
        st.markdown(f"#### Question {quiz['step'] + 1} of 3")
        
        # BULK GENERATION OPTIMIZATION
        if not quiz["questions"]:
            with st.spinner("Generating all security challenges locally for maximum speed..."):
                # Get context for 3 different questions
                user_txns = df[(df['first'] == first_name) & (df['last'] == last_name)]
                if len(user_txns) < 3:
                     st.warning("Insufficient transaction history for a secure 3-step audit.")
                     return

                # SELECT 3 SPECIFIC GROUND-TRUTH TARGETS
                targets = user_txns.sample(3).to_dict('records')
                target_features = []
                for t in targets:
                    f = random.choice(['merchant', 'amt', 'category'])
                    target_features.append({"feature": f, "value": t[f]})
                
                features_ctx = "\n".join([f"- Question {i+1}: Feature is '{tf['feature']}', Correct Value is '{tf['value']}'" for i, tf in enumerate(target_features)])
                
                prompt = f"""Generate EXACTLY 3 identity verification questions for {selected_user}.
You MUST use these specific Ground Truth values for the CORRECT answers:
{features_ctx}

STRICT RULES:
1. Return EXACTLY 3 questions.
2. For each, the CORRECT option MUST contain the Ground Truth value provided above.
3. Provide 3 incorrect distractor options for each.
4. Format:
[Question 1]
Question: [Text]
A) [Option Text]
B) [Option Text]
C) [Option Text]
D) [Option Text]
Correct: [Letter]
[End Question]
"""
                raw_q_bulk = get_llm_via_api(prompt)
                
                # Resilient bulk parser
                blocks = re.split(r'\[Question \d\]', raw_q_bulk)
                for i, block in enumerate(blocks):
                    if len(quiz["questions"]) >= 3:
                        break
                        
                    if "Question:" in block:
                        lines = [l.strip() for l in block.split("\n") if l.strip()]
                        q_text = next((l.replace("Question:", "").strip() for l in lines if l.startswith("Question:")), "Identity Check")
                        all_opts = [l for l in lines if re.match(r'^[A-D]\)', l)]
                        opts = all_opts[:4]
                        
                        while len(opts) < 4:
                            letter = chr(65 + len(opts))
                            opts.append(f"{letter}) Other activity")

                        # Answer parsing
                        correct_line = next((l for l in lines if l.startswith("Correct:")), "Correct: A")
                        after_colon = correct_line.split(":")[-1]
                        correct_search = re.search(r'[A-D]', after_colon.upper())
                        correct_letter = correct_search.group(0) if correct_search else "A"
                        
                        # Store Ground Truth for verification
                        quiz["questions"].append({
                            "text": q_text,
                            "options": opts,
                            "correct_letter": correct_letter,
                            "ground_truth": target_features[len(quiz["questions"])]["value"]
                        })
                
                # Fallback if generation failed
                if not quiz["questions"]:
                    st.error("Failed to generate challenges. Please retry.")
                    return
                st.rerun()

        # Display current question
        current_q = quiz["questions"][quiz["step"]]
        st.markdown(f"**{current_q['text']}**")
        user_choice = st.radio("Choose your answer:", [o[0] for o in current_q["options"]], 
                              format_func=lambda x: next(o for o in current_q["options"] if o.startswith(x)))
        
        if st.button("Submit Answer"):
            # GROUND TRUTH VERIFICATION: Check if the selected option text contains the actual data
            selected_option_text = next(o for o in current_q["options"] if o.startswith(user_choice))
            ground_truth = str(current_q["ground_truth"])
            
            # Case-insensitive check to ensure the user picked the real data
            is_correct = ground_truth.lower() in selected_option_text.lower()
            
            quiz["user_responses"].append({
                "question": current_q["text"],
                "your_answer": selected_option_text,
                "ground_truth": ground_truth,
                "is_correct": is_correct
            })

            if is_correct:
                 quiz["score"] += 1
                 quiz["last_result"] = f"✅ Verified: {ground_truth} matches records."
            else:
                 quiz["last_result"] = f"❌ Verification Failed. (Evidence did not match history)."
            
            quiz["step"] += 1
            if quiz["step"] >= 3:
                quiz["completed"] = True
            st.rerun()

        if quiz["last_result"]:
            st.toast(quiz["last_result"])

    else:
        # Quiz completed
        st.success(f"Authentication Process Completed for {selected_user}!")
        st.markdown(f"""
        <div class='metric-card' style='padding: 2rem;'>
            <h3>Verification Score</h3>
            <div class='value' style='color:{"#16a34a" if quiz["score"] >= 2 else "#dc2626"};'>{quiz["score"]}/3</div>
            <p>{ "Identity Verified ✅" if quiz["score"] >= 2 else "Verification Failed ❌" }</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 📑 Verification Audit Log")
        # Build a nice table for the summary
        summary_data = []
        for r in quiz["user_responses"]:
            summary_data.append({
                "Security Challenge": r["question"],
                "Your Response": r["your_answer"],
                "Ground Truth (Actual Records)": r["ground_truth"],
                "Status": "✅ PASS" if r["is_correct"] else "❌ FAIL"
            })
        
        st.table(pd.DataFrame(summary_data))
        
        if st.button("Restart Authentication"):
            del st.session_state.auth_quiz
            st.rerun()

# ---------------------------------------------------------------------------
# Tab 4: CFPB Market Intelligence
# ---------------------------------------------------------------------------
def render_cfpb_tab(df):
    st.markdown("### 🔍 CFPB Credit Card Intelligence")
    if df.empty:
        st.warning("CFPB dataset missing or empty.")
        return

    m1, m2 = st.columns([1.2, 1])
    with m2:
        st.markdown("#### Complaint Trends")
        top_cos = df['Company'].value_counts().head(8).reset_index()
        fig = px.bar(top_cos, x='count', y='Company', orientation='h', template='plotly_white', 
                     color='count', color_continuous_scale='Cividis', title="Volume by Financial Institution")
        st.plotly_chart(apply_accessible_theme(fig), use_container_width=True)

    with m1:
        st.markdown("#### Complaint Analysis")
        st.info("The complaint database provides insight into consumer pain points and institutional response trends.")

        st.markdown("#### Geographic Distribution")
        state_counts = df['State'].value_counts().head(10).reset_index()
        fig2 = px.pie(state_counts, values='count', names='State', hole=0.5, template='plotly_white',
                      color_discrete_sequence=px.colors.qualitative.Antique, title="Top States by Complaint Density")
        st.plotly_chart(apply_accessible_theme(fig2), use_container_width=True)

    st.divider()
    st.markdown("#### 💬 Ask Local RAG")
    query = st.text_input("Search context for fraud patterns or policy details:", placeholder="e.g. Find high-value travel anomalies...")
    if st.button("Query Knowledge Base") and query:
        with st.spinner("Querying RAG via API..."):
            try:
                resp = requests.post(f"{API_BASE_URL}/api/rag/query", json={"query": query, "n_results": 5}, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    if data["results"]:
                        for r in data["results"]:
                            st.markdown(f"""
                            <div class='rag-answer' style='margin-bottom:1rem; border-color: rgba(99, 102, 241, 0.3);'>
                                <strong>[{r['confidence']:.1%} Relevance]</strong><br>{r['text']}
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.warning("No relevant context found.")
                else:
                    st.error(f"API Error: {resp.status_code}")
            except requests.ConnectionError:
                st.error("🔴 Cannot reach API backend. Please start it with: `uvicorn api.main:app --port 8000`")



# ---------------------------------------------------------------------------
# Tab 5: Agentic Analyst
# ---------------------------------------------------------------------------
def render_agent_tab():
    st.markdown("### 🤖 Agentic Security Analyst")
    st.info("The GuardAgent is an autonomous analyst powered by a FastAPI microservice. It uses tool-based reasoning to investigate systemic risks.")

    user_query = st.text_area("Investigation Request:", 
                             placeholder="e.g. Investigate risk for USER_0. What are the top 3 high-risk transactions across the system?",
                             height=150)
    
    if st.button("Initiate Investigation") and user_query:
        with st.spinner("Agent is reasoning via API backend..."):
            try:
                resp = requests.post(f"{API_BASE_URL}/api/agent/investigate", json={"query": user_query}, timeout=120)
                if resp.status_code == 200:
                    result = resp.json()
                    
                    # Show steps/history
                    if result.get("actions"):
                        with st.expander("🔍 Trace: Agent Reasoning Steps", expanded=True):
                            for a in result["actions"]:
                                st.markdown(f"**Step {a['step']}:** Calling `{a['tool']}` with args `{a['args']}`")
                                if a.get('result'):
                                    st.code(a['result'], language="json")
                    
                    # Show final report
                    st.markdown("<div class='rag-answer'>", unsafe_allow_html=True)
                    st.markdown("#### 🛡️ GuardAgent Report")
                    st.markdown(result["answer"])
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.error(f"API Error: {resp.status_code} — {resp.text}")
            except requests.ConnectionError:
                st.error("🔴 Cannot reach API backend. Please start it with: `uvicorn api.main:app --port 8000`")

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def main():
    render_sidebar()

    st.markdown("""
    <div class="main-header">
        <h1>Veriscan Dashboard</h1>
        <p>Real-time Fraud Prevention & Consumer Compliance Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["📊 Market Dash", "🚨 AI Fraud ML", "🔑 Auth Challenges", "🔍 CFPB Market Intel", "🤖 Agentic Analyst"])
    
    fraud_df = load_fraud_data()
    cfpb_df = load_cfpb_data()
    model, encoders = load_fraud_model()

    with tabs[0]: render_dashboard_tab(fraud_df)
    with tabs[1]: render_prediction_tab(fraud_df, model, encoders)
    with tabs[2]: render_auth_tab(fraud_df)
    with tabs[3]: render_cfpb_tab(cfpb_df)
    with tabs[4]: render_agent_tab()

if __name__ == "__main__":
    main()
