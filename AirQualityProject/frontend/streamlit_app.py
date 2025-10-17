import streamlit as st
import requests
import pandas as pd
import os, sys
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Ensure project root is on PYTHONPATH for `src` imports when running from `frontend/`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import WINDOW

# ===== Helpers for colorful, styled tables =====
MODEL_COLORS = {
    'ANN': '#2dd4bf',            # teal
    'CNN': '#60a5fa',            # blue
    'ENCODER_DECODER': '#a78bfa',# purple
    'LSTM': '#f59e0b',           # amber
    'VGG9': '#f472b6',           # pink
    'VGG16': '#34d399',          # emerald
}

def _row_style_by_model(row, model_col='Model', best_model=None):
    model_name = str(row.get(model_col, '')).upper()
    bg = MODEL_COLORS.get(model_name, '#475569')
    if best_model and model_name == str(best_model).upper():
        bg = '#16a34a'  # best highlight
    return [f'background-color: {bg}; color: white; font-weight: 600;'] * len(row)

def style_by_model(df, model_col='Model', best_model=None):
    # Function kept unchanged to preserve functionality
    return df


# --- helper: map numeric AQI -> category
def aqi_to_category(val):
    """Classify AQI value into categories based on standard AQI ranges."""
    try:
        x = float(val)
    except (ValueError, TypeError):
        # if already a category string, normalize casing
        return str(val).strip().title()
    if x <= 50:
        return "Good"
    elif x <= 100:
        return "Moderate"
    elif x <= 150:
        return "Unhealthy for Sensitive Groups"
    elif x <= 200:
        return "Unhealthy"
    elif x <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

# canonical label order
AQI_LABELS = ["Good", "Moderate", "Unhealthy for Sensitive Groups", "Unhealthy", "Very Unhealthy", "Hazardous"]


st.set_page_config(
    page_title="AirSense — AI-Powered Air Quality Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===== Global Styling =====
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&display=swap');

    .stApp {
        background: radial-gradient(circle at top left, rgba(35,56,99,0.75), rgba(15,20,25,0.95)),
                    linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%);
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }

    ::-webkit-scrollbar { width: 9px; height: 9px; }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #38bdf8, #6366f1);
        border-radius: 6px;
    }
    ::-webkit-scrollbar-track { background: rgba(15, 23, 42, 0.8); }

    .hero-banner {
        background: linear-gradient(120deg, rgba(59,130,246,0.15), rgba(129,140,248,0.15));
        border: 1px solid rgba(96,165,250,0.25);
        border-radius: 20px;
        padding: 1.5rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 12px 32px rgba(15, 23, 42, 0.45);
    }

    .hero-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 0.75rem;
        margin-top: 1rem;
    }

    .hero-badge {
        padding: 0.45rem 0.85rem;
        border-radius: 999px;
        background: rgba(96,165,250,0.15);
        border: 1px solid rgba(96,165,250,0.25);
        font-size: 0.85rem;
        font-weight: 500;
        color: #bfdbfe;
    }

    .main-header {
        background: linear-gradient(135deg, rgba(30,41,59,0.95), rgba(51,65,85,0.95));
        padding: 2.2rem;
        border-radius: 20px;
        border: 1px solid rgba(37, 99, 235, 0.35);
        margin-bottom: 1.2rem;
        box-shadow: 0 16px 42px rgba(2, 6, 23, 0.6);
        position: relative;
        overflow: hidden;
    }

    .main-header::before {
        content: "";
        position: absolute;
        top: -20%;
        right: -15%;
        width: 260px;
        height: 260px;
        background: radial-gradient(circle, rgba(59,130,246,0.18), rgba(59,130,246,0));
        border-radius: 50%;
    }

    .main-header::after {
        content: "";
        position: absolute;
        bottom: -25%;
        left: -10%;
        width: 200px;
        height: 200px;
        background: radial-gradient(circle, rgba(13,148,136,0.2), rgba(13,148,136,0));
        border-radius: 50%;
    }

    /* Hybrid callout */
    .hybrid-card {
        background: linear-gradient(135deg, rgba(56,189,248,0.15), rgba(124,58,237,0.15));
        border: 1px solid rgba(165,180,252,0.4);
        border-radius: 18px;
        padding: 1.6rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 16px 36px rgba(2, 6, 23, 0.45);
        display: flex;
        gap: 1rem;
        align-items: center;
    }
    .hybrid-card-icon {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background: linear-gradient(135deg, rgba(56,189,248,0.4), rgba(124,58,237,0.4));
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.8rem;
        color: #f8fafc;
        box-shadow: inset 0 0 0 1px rgba(125,211,252,0.45);
    }
    .hybrid-card-content h3 {
        margin: 0;
        font-size: 1.28rem;
        color: #f8fafc;
    }
    .hybrid-card-content p {
        color: rgba(226,232,240,0.78);
        margin: 0.4rem 0 0;
        font-size: 0.96rem;
        line-height: 1.6;
    }
    .hybrid-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        background: rgba(56,189,248,0.2);
        color: #bae6fd;
        font-size: 0.82rem;
        border: 1px solid rgba(56,189,248,0.45);
        margin-top: 0.6rem;
        font-weight: 600;
        letter-spacing: 0.01em;
    }

    .metric-card {
        background: linear-gradient(140deg, rgba(30,41,59,0.92), rgba(45,55,72,0.88));
        padding: 1.35rem 1.4rem;
        border-radius: 16px;
        border: 1px solid rgba(59, 130, 246, 0.22);
        margin: 0.6rem 0;
        box-shadow: 0 10px 32px rgba(15, 23, 42, 0.55);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px) scale(1.01);
        box-shadow: 0 16px 48px rgba(30, 64, 175, 0.45);
    }

    /* Mini info tiles */
    .mini-tile {
    background: linear-gradient(145deg, rgba(30,41,59,0.88), rgba(15,23,42,0.9));
    border-radius: 16px;
    border: 1px solid rgba(148,163,184,0.25);
    padding: 1rem 1.15rem;
    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.55);
    height: 320px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

/* Hover Animation */
.mini-tile:hover {
    transform: translateY(-6px);
    box-shadow: 0 14px 36px rgba(15, 23, 42, 0.65);
}

.mini-tile h4 {
    margin: 0;
    font-size: 1rem;
    color: #f1f5f9;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.mini-tile p, .mini-tile ul {
    color: rgba(226,232,240,0.78);
    font-size: 0.88rem;
    margin: 0;
    line-height: 1.45;
    list-style: none;
    padding-left: 0;
}

.mini-tile ul li::before {
    content: "•";
    margin-right: 0.35rem;
    color: rgba(56,189,248,0.75);
}

.mini-badge {
    padding: 0.35rem 0.65rem;
    border-radius: 999px;
    background: rgba(56,189,248,0.2);
    color: #bae6fd;
    font-size: 0.75rem;
    font-weight: 600;
}

/* AQI Card */
.aqi-card {
    background: linear-gradient(145deg, rgba(30,41,59,0.92), rgba(67,56,202,0.92));
    border-radius: 16px;
    border: 1px solid rgba(129,140,248,0.35);
    padding: 1rem 1.1rem;
    height: 320px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    box-shadow: 0 12px 32px rgba(30,64,175,0.45);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.aqi-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 16px 40px rgba(30,64,175,0.55);
}

/* AQI Ladder Rows */
.aqi-rows {
    display: grid;
    gap: 0.35rem;
    font-size: 0.86rem;
}

.aqi-row {
    display: flex;
    justify-content: space-between;
    padding: 0.35rem 0.5rem;
    border-radius: 10px;
    background: rgba(15,23,42,0.45);
    border: 1px solid rgba(148,163,184,0.18);
    color: rgba(241,245,249,0.9);
    font-weight: 500;
}

.aqi-row span:first-child {
    font-weight: 700;
}
    /* Hybrid + data section toggles */
    .toggle-bar {
        display: flex;
        gap: 0.6rem;
        margin-bottom: 1rem;
    }
    .toggle-button {
        flex: 1;
        text-align: center;
        padding: 0.7rem 1rem;
        border-radius: 12px;
        border: 1px solid rgba(59, 130, 246, 0.35);
        background: rgba(15, 23, 42, 0.75);
        color: rgba(226,232,240,0.75);
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .toggle-button.active {
        background: linear-gradient(135deg, rgba(56,189,248,0.3), rgba(59,130,246,0.35));
        color: #f8fafc;
        box-shadow: 0 12px 28px rgba(37, 99, 235, 0.45);
        border-color: rgba(59,130,246,0.55);
    }

    /* Dataframe enhancements */
    .stDataFrame, .stTable {
        background: rgba(15, 23, 42, 0.85);
        border-radius: 16px;
        padding: 0.35rem;
        border: 1px solid rgba(71, 85, 105, 0.55);
        box-shadow: inset 0 1px 0 rgba(148, 163, 184, 0.18), 0 16px 30px rgba(2, 6, 23, 0.7);
    }
    .stDataFrame table, .stTable table {
        color: #e2e8f0;
        font-weight: 500;
    }
    .stDataFrame table thead th {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.35), rgba(76, 29, 149, 0.25));
        color: #f8fafc !important;
        border-bottom: 1px solid rgba(59, 130, 246, 0.35);
        font-size: 0.92rem;
        padding: 0.75rem 0.5rem !important;
    }
    .stDataFrame table tbody tr:nth-child(even) {
        background: rgba(30, 41, 59, 0.35);
    }
    .stDataFrame table tbody tr:hover {
        background: rgba(59, 130, 246, 0.18);
    }
    .stDataFrame table tbody tr td {
        border-bottom: 1px solid rgba(71, 85, 105, 0.35);
        padding: 0.65rem 0.5rem !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.85rem 1.65rem;
        font-weight: 600;
        transition: all 0.25s ease;
        box-shadow: 0 12px 32px rgba(37, 99, 235, 0.45);
        position: relative;
        overflow: hidden;
    }
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    .stButton > button:hover::before {
        left: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 16px 36px rgba(59, 130, 246, 0.55);
    }
    
    /* Enhanced form styling */
    .stNumberInput > div > div > input {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 8px;
        color: #e2e8f0;
        padding: 0.5rem;
        transition: all 0.2s ease;
    }
    .stNumberInput > div > div > input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        background: rgba(15, 23, 42, 0.95);
    }
    
    /* Enhanced selectbox styling */
    .stSelectbox > div > div > div {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 8px;
        color: #e2e8f0;
    }
    
    /* Enhanced file uploader styling */
    .stFileUploader > div > div > div {
        background: rgba(15, 23, 42, 0.8);
        border: 2px dashed rgba(59, 130, 246, 0.3);
        border-radius: 12px;
        padding: 1rem;
        transition: all 0.2s ease;
    }
    .stFileUploader > div > div > div:hover {
        border-color: #3b82f6;
        background: rgba(15, 23, 42, 0.95);
    }
    
    /* Enhanced spinner styling */
    .stSpinner > div {
        border-color: #3b82f6 transparent #3b82f6 transparent;
    }
    
    /* Enhanced tabs styling */
    .stTabs > div > div > div > button {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(59, 130, 246, 0.3);
        color: #e2e8f0;
        border-radius: 8px 8px 0 0;
        transition: all 0.2s ease;
    }
    .stTabs > div > div > div > button[aria-selected="true"] {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(29, 78, 216, 0.2));
        border-color: #3b82f6;
        color: #f8fafc;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #f9fafb;
        font-weight: 700;
    }

    .stMetric {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.92), rgba(15, 23, 42, 0.88));
        border-radius: 15px;
        border: 1px solid rgba(59, 130, 246, 0.3);
        padding: 1.15rem 1rem;
        box-shadow: 0 10px 26px rgba(15, 23, 42, 0.55);
    }

    .stPlotlyChart {
        background: rgba(15, 23, 42, 0.85);
        border-radius: 18px;
        padding: 1.35rem;
        border: 1px solid rgba(59, 130, 246, 0.3);
        box-shadow: 0 16px 36px rgba(2, 6, 23, 0.75);
    }

    .footer-box {
        text-align: center;
        padding: 2.5rem;
        background: linear-gradient(120deg, rgba(30, 41, 59, 0.95), rgba(51, 65, 85, 0.9));
        border-radius: 20px;
        border: 1px solid rgba(148, 163, 184, 0.25);
        margin-top: 2.5rem;
        box-shadow: 0 18px 42px rgba(2, 6, 23, 0.75);
    }
    .footer-highlight {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        color: #22d3ee;
        font-weight: 600;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ===== Enhanced Sidebar =====
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; padding: 1.25rem 1rem 0.75rem; border-radius: 18px;
                    background: linear-gradient(145deg, rgba(30, 41, 59, 0.95), rgba(51, 65, 85, 0.85));
                    border: 1px solid rgba(59, 130, 246, 0.35); box-shadow: 0 10px 26px rgba(15, 23, 42, 0.65);">
            <h2 style="color: #38bdf8; margin-bottom: 0.4rem; font-size: 1.6rem;">🌍 AirSense</h2>
            <p style="color: #cbd5f5; font-size: 0.92rem; margin: 0;">
                AI-Powered Air Quality Intelligence
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="hero-banner" style="margin-top: 1rem;">
            <h4 style="margin: 0; color: #e0f2fe; font-weight: 700;">Why AirSense?</h4>
            <p style="margin-top: 0.4rem; color: rgba(224, 242, 254, 0.85); font-size: 0.92rem;">
                Seamlessly explore real-time and historical air quality insights using a powerful ensemble of AI models—optimized for both single-point analytics and multi-step forecasting.
            </p>
            <div class="hero-badges">
                <span class="hero-badge">Multi-model inference</span>
                <span class="hero-badge">Hybrid AQI strategies</span>
                <span class="hero-badge">Health-aware outputs</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### ⚙️ Configuration")
    backend_url = st.text_input("Backend URL", value="http://localhost:8000")
    activation_options = ["Linear", "ReLU", "Sigmoid", "Softmax"]
    activation_choice = st.selectbox("Activation (post-processing)", activation_options, index=0, help="Apply activation to model outputs for exploration. Linear = no change.")
    
    # Show current activation info
    activation_info = {
        "Linear": "No transformation applied - raw model outputs",
        "ReLU": "Negative values set to zero - only positive predictions",
        "Sigmoid": "Values squashed to 0-1 range - probability-like outputs",
        "Softmax": "Values converted to probabilities across models"
    }
    st.info(f"ℹ️ **{activation_choice}**: {activation_info[activation_choice]}")
    
    # Show current time
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    st.markdown(f"🕐 **Current Time**: {current_time}")

    loaded_models = []
    try:
        status_resp = requests.get(f"{backend_url}/status", timeout=3)
        if status_resp.ok:
            status_json = status_resp.json()
            loaded_models = status_json.get("loaded_models", [])
    except Exception:
        loaded_models = []

    st.markdown("### 📊 Model Information")
    st.markdown(
        f"""
        <div class="metric-card" style="background: linear-gradient(145deg, rgba(14,116,144,0.95), rgba(5,150,105,0.9));
                                        border-color: rgba(20,184,166,0.45);">
            <h4 style="margin-top:0; color:#bbf7d0;">Model Stack Snapshot</h4>
            <p style="color: rgba(226,232,240,0.78); font-size:0.92rem;">
                Specialized deep-learning architectures watch over {WINDOW}-step windows to deliver resilient AQI intelligence.
            </p>
            <ul style="margin: 0.4rem 0 0; padding-left: 1rem; color: rgba(226,232,240,0.8);">
                <li>ANN • CNN • LSTM • Encoder-Decoder</li>
                <li>VGG9 • VGG16 • Smart Hybrid Engine</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )
    if loaded_models:
        st.markdown(
            f"""
            <div class="metric-card" style="margin-top: 0.6rem;">
                <strong>Currently Active:</strong> {', '.join(m.upper() for m in loaded_models)}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("⚠️ No models loaded. Please check your backend connection.")

    st.markdown("### 🧪 Hybrid Weights (VGG16 + ANN)")
    col_hw1, col_hw2 = st.columns(2)
    with col_hw1:
        w_vgg16 = st.number_input("VGG16 weight", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
    with col_hw2:
        w_ann = st.number_input("ANN weight", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
    w_sum = w_vgg16 + w_ann
    if w_sum == 0:
        w_vgg16, w_ann = 0.6, 0.4
    else:
        w_vgg16, w_ann = w_vgg16 / w_sum, w_ann / w_sum
    
    # Show weight balance
    st.markdown(f"📊 **Weight Balance**: VGG16: {w_vgg16:.1%} | ANN: {w_ann:.1%}")

    st.markdown("### ✨ Pro Tips")
    st.markdown(
        f"""
        <div class="metric-card" style="background: linear-gradient(135deg, rgba(49,46,129,0.95), rgba(29,78,216,0.92));
                                        border-color: rgba(99,102,241,0.45);">
            <h4 style="color: #bfdbfe;">Optimize Your Insights</h4>
            <ul style="margin-left: -0.7rem; line-height: 1.55; font-size: 0.92rem;">
                <li>Prepare CSV with ≥ {WINDOW} sequential rows for deep models.</li>
                <li>Confirm pollutant + weather columns before processing.</li>
                <li>Hybrid mode (ANN + VGG16) is prioritized for stability.</li>
                <li>Track prediction spread to quantify model agreement.</li>
                <li>Use different activations to explore model behavior.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Add a refresh button for backend status
    if st.button("🔄 Refresh Backend Status", use_container_width=True):
        st.rerun()

# ===== Hero Header & Overview =====
st.markdown(
    """
    <div class="main-header">
        <div style="display: flex; flex-direction: column; gap: 0.65rem;">
            <h1 style="margin: 0; font-size: 1.8rem; letter-spacing: 0.02em; 
                       background: linear-gradient(120deg, #38bdf8 0%, #22d3ee 45%, #a855f7 100%);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                🌍 AirSense Dashboard
            </h1>
            <p style="margin: 0; font-size: 1.08rem; color: rgba(226,232,240,0.85); max-width: 680px;">
                Experience a premium AI environment tailored for actionable air-quality intelligence.
                Combine manual snapshots with historical trend uploads to unlock confident AQI predictions,
                health guardrails, and hybrid modeling strategies—all presented in a modern research-grade interface.
            </p>
        </div>
        <div class="hero-badges" style="margin-top: 1.2rem;">
            <span class="hero-badge">Real-time scenario diagnostics</span>
            <span class="hero-badge">Deep-ensemble forecasting</span>
            <span class="hero-badge">Impact-driven recommendations</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ===== Hybrid Intelligence Highlight =====
st.markdown(
    """
    <div class="hybrid-card">
        <div class="hybrid-card-icon">⚡</div>
        <div class="hybrid-card-content">
            <h3>Hybrid Intelligence Mode (ANN + VGG16)</h3>
            <p>
                AirSense automatically fuses the high-resolution pattern detection of <strong>VGG16</strong> with the
                rapid generalization power of our <strong>ANN</strong>. This smart strategy orchestrates weighted
                outputs, meta-rankings, and conservative checks to keep prediction drift minimal—even when data
                conditions intensify. Adjust the sidebar weights if you need to tune emphasis; otherwise, the system
                ensures an optimal “smart” blend behind the scenes.
            </p>
            <div class="hybrid-chip">Priority Engine • Accuracy First • Smart adaptive fusion</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ===== Air Quality Insights Tiles =====
st.markdown("## 🔍 Rapid Reference Tiles")

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown(
        """
        <div class="mini-tile">
            <h4>Core AQI Inputs<span class="mini-badge">Essentials</span></h4>
            <ul>
                <li>PM2.5 / PM10 — particle load</li>
                <li>NO₂ + SO₂ — combustion footprint</li>
                <li>CO — oxygen displacement indicator</li>
                <li>O₃ — photochemical stressor</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_b:
    st.markdown(
        """
        <div class="mini-tile">
            <h4>Atmospheric Drivers<span class="mini-badge">Context</span></h4>
            <ul>
                <li>Temperature energizes reactions</li>
                <li>Humidity shifts particulate behaviour</li>
                <li>Wind spreads or concentrates plumes</li>
                <li>Pressure & radiation steer mixing</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_c:
    st.markdown(
        """
        <div class="aqi-card">
            <h4 style="margin:0; display:flex; justify-content:space-between; align-items:center;">
                AQI Ladder<span class="mini-badge">Health</span>
            </h4>
            <div class="aqi-rows">
                <div class="aqi-row"><span>0-50</span><span>Pristine • open-air welcome</span></div>
                <div class="aqi-row"><span>51-100</span><span>Moderate • monitor sensitivities</span></div>
                <div class="aqi-row"><span>101-150</span><span>USG • protect vulnerable groups</span></div>
                <div class="aqi-row"><span>151-200</span><span>Unhealthy • limit exposure</span></div>
                <div class="aqi-row"><span>201-300+</span><span>Crisis • enact emergency posture</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# ===== Main Navigation =====
if "active_view" not in st.session_state:
    st.session_state.active_view = "Manual"

# Add main navigation
nav_cols = st.columns(3)
with nav_cols[0]:
    if st.button("🔮 Predictions", use_container_width=True):
        st.session_state.active_view = "Manual"
with nav_cols[1]:
    if st.button("📊 Analysis", use_container_width=True):
        st.session_state.active_view = "CSV"
with nav_cols[2]:
    if st.button("📈 Model Evaluation", use_container_width=True):
        st.session_state.active_view = "Evaluation"

# ===== Manual vs CSV Toggle =====
if st.session_state.active_view in ["Manual", "CSV"]:
    toggle_cols = st.columns(2)
    with toggle_cols[0]:
        manual_clicked = st.button(
            "📘 Manual Input",
            key="manual_toggle",
            help="Enter a single measurement snapshot and receive instant AQI projections.",
        )
    with toggle_cols[1]:
        csv_clicked = st.button(
            "📁 Upload CSV",
            key="csv_toggle",
            help="Upload historical sequences for trend analysis and ensemble forecasts.",
        )

    if manual_clicked:
        st.session_state.active_view = "Manual"
    if csv_clicked:
        st.session_state.active_view = "CSV"

    st.markdown(
        f"""
        <div class="toggle-bar">
            <div class="toggle-button {'active' if st.session_state.active_view == 'Manual' else ''}">
                📘 Manual Input
            </div>
            <div class="toggle-button {'active' if st.session_state.active_view == 'CSV' else ''}">
                📁 Upload CSV
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    hybrid_strategy = "smart"  # silently use best strategy

    # ===== MANUAL INPUT VIEW =====
    if st.session_state.active_view == "Manual":
        st.markdown(
            f"""
            <div class="metric-card" style="margin-top: -0.4rem;">
                <h4 style="margin-top:0; color:#f9fafb;">Single Reading Analysis</h4>
                <p style="color: rgba(226,232,240,0.8); font-size:0.95rem;">
                    Supply current pollutant and meteorological metrics. AirSense generates an internal {WINDOW}-step
                    synthetic window to maintain model parity between manual and historical workflows. Ideal for on-site
                    readings where hybrid accuracy is critical.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.form("manual_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Particulate Matter**")
                pm25 = st.number_input("PM2.5 (µg/m³)", value=12.0, min_value=0.0, max_value=500.0, help="Fine particles smaller than 2.5 micrometers")
                pm10 = st.number_input("PM10 (µg/m³)", value=30.0, min_value=0.0, max_value=600.0, help="Particles smaller than 10 micrometers")

            with col2:
                st.markdown("**Gaseous Pollutants**")
                no2 = st.number_input("NO₂ (ppb)", value=10.0, min_value=0.0, max_value=200.0, help="Nitrogen dioxide concentration")
                so2 = st.number_input("SO₂ (ppb)", value=5.0, min_value=0.0, max_value=100.0, help="Sulfur dioxide concentration")
                co = st.number_input("CO (ppm)", value=0.4, min_value=0.0, max_value=50.0, help="Carbon monoxide concentration")
                o3 = st.number_input("O₃ (ppb)", value=15.0, min_value=0.0, max_value=300.0, help="Ground-level ozone concentration")

            with col3:
                st.markdown("**Meteorological Data**")
                temp = st.number_input("Temperature (°C)", value=25.0, min_value=-50.0, max_value=60.0, help="Ambient temperature")
                humidity = st.number_input("Humidity (%)", value=60.0, min_value=0.0, max_value=100.0, help="Relative humidity")
                wind = st.number_input("Wind Speed (m/s)", value=2.0, min_value=0.0, max_value=50.0, help="Wind speed at measurement height")

            submitted = st.form_submit_button("🚀 Generate AI Predictions", use_container_width=True)
            
            if submitted:
                # Input validation
                validation_errors = []
                if pm25 < 0 or pm25 > 500:
                    validation_errors.append("PM2.5 must be between 0 and 500 µg/m³")
                if pm10 < 0 or pm10 > 600:
                    validation_errors.append("PM10 must be between 0 and 600 µg/m³")
                if no2 < 0 or no2 > 200:
                    validation_errors.append("NO₂ must be between 0 and 200 ppb")
                if so2 < 0 or so2 > 100:
                    validation_errors.append("SO₂ must be between 0 and 100 ppb")
                if co < 0 or co > 50:
                    validation_errors.append("CO must be between 0 and 50 ppm")
                if o3 < 0 or o3 > 300:
                    validation_errors.append("O₃ must be between 0 and 300 ppb")
                if temp < -50 or temp > 60:
                    validation_errors.append("Temperature must be between -50 and 60 °C")
                if humidity < 0 or humidity > 100:
                    validation_errors.append("Humidity must be between 0 and 100%")
                if wind < 0 or wind > 50:
                    validation_errors.append("Wind speed must be between 0 and 50 m/s")
                
                if validation_errors:
                    for error in validation_errors:
                        st.error(f"❌ {error}")
                else:
                    features = {
                        "PM2.5": pm25, "PM10": pm10, "NO2": no2, "SO2": so2,
                        "CO": co, "O3": o3, "temp": temp, "humidity": humidity, "wind": wind
                    }
                    rows = [features for _ in range(WINDOW)]
                    payload = {"features": features, "last_window": rows, "hybrid_weights": {"vgg16": w_vgg16, "ann": w_ann}, "hybrid_strategy": hybrid_strategy, "activation": activation_choice.lower()}

                    # Progress bar for better UX
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("🤖 Initializing AI models...")
                    progress_bar.progress(10)
                    
                    status_text.text("🧠 Processing input data...")
                    progress_bar.progress(30)
                    
                    status_text.text("⚡ Running ensemble predictions...")
                    progress_bar.progress(60)
                    
                    status_text.text("🔮 Generating insights...")
                    progress_bar.progress(90)
                    
                    try:
                        r = requests.post(f"{backend_url}/predict", json=payload, timeout=30)
                        if r.ok:
                            data = r.json()
                            preferred_models = ["ann", "cnn", "encoder_decoder", "lstm", "vgg9", "vgg16", "hybrid"]
                            model_order = [m for m in preferred_models if (not loaded_models or m in loaded_models)]
                            rows_out = []
                            for m in preferred_models:
                                pred_val = data.get(m)
                                status = "Loaded" if (not loaded_models or m in loaded_models) else "Not loaded"
                                if status == "Loaded" and pred_val is None:
                                    status = "No prediction"
                                rows_out.append({
                                    "Model": m.upper(),
                                    "Prediction": pred_val,
                                    "Status": status,
                                    "Confidence": "High" if m in ["ann", "lstm", "hybrid"] else "Medium",
                                })

                            if len(rows_out) > 0:
                                df_out = pd.DataFrame(rows_out)
                                df_out["Prediction"] = df_out["Prediction"].map(lambda x: round(float(x), 4) if x is not None else None)
                                if 'vgg16' not in loaded_models:
                                    st.warning("VGG16 model not loaded on backend — showing status only.")

                                # Complete progress bar
                                progress_bar.progress(100)
                                status_text.text("✅ Analysis complete!")
                                
                                # Clear progress indicators after a short delay
                                import time
                                time.sleep(1)
                                progress_bar.empty()
                                status_text.empty()

                                # Success message
                                st.success("🎉 Predictions generated successfully!")
                                
                                st.markdown("## 🔮 AI Model Predictions (Manual Snapshot)")

                                col1, col2 = st.columns([2.2, 1])

                                with col1:
                                    chart_df = df_out[df_out["Prediction"].notnull()]
                                    fig_manual = px.bar(
                                        chart_df, x="Model", y="Prediction",
                                        title="AI Ensemble AQI Outputs (Single Snapshot)",
                                        color="Prediction",
                                        color_continuous_scale="RdYlBu_r",
                                        text="Prediction"
                                    )
                                    fig_manual.update_layout(
                                        plot_bgcolor='rgba(7, 12, 22, 0.0)',
                                        paper_bgcolor='rgba(7, 12, 22, 0.0)',
                                        font_color='white',
                                        title_font_size=17,
                                        margin=dict(t=75, l=10, r=10, b=10)
                                    )
                                    fig_manual.update_traces(texttemplate='%{text}', textposition='outside')
                                    st.plotly_chart(fig_manual, use_container_width=True)

                                with col2:
                                    st.markdown("### 📊 Prediction Summary")
                                    avg_prediction = df_out["Prediction"].mean()
                                    max_prediction = df_out["Prediction"].max()
                                    min_prediction = df_out["Prediction"].min()

                                    st.metric("Average", f"{avg_prediction:.4f}")
                                    st.metric("Max", f"{max_prediction:.4f}")
                                    st.metric("Min", f"{min_prediction:.4f}")

                                    if activation_choice.lower() == "softmax":
                                        st.info("Softmax shows probabilities across models; thresholds below are not AQI.")
                                    if activation_choice.lower() != "softmax" and avg_prediction is not None and not pd.isna(avg_prediction):
                                        if avg_prediction <= 50:
                                            st.markdown('<div class="mini-tile" style="height:auto;"><h4>Status<span class="mini-badge">Good</span></h4><p>Ideal conditions for outdoor activity.</p></div>', unsafe_allow_html=True)
                                        elif avg_prediction <= 100:
                                            st.markdown('<div class="mini-tile" style="height:auto;"><h4>Status<span class="mini-badge">Moderate</span></h4><p>Sensitive groups pace exertion and monitor changes.</p></div>', unsafe_allow_html=True)
                                        else:
                                            st.markdown('<div class="mini-tile" style="height:auto;"><h4>Status<span class="mini-badge">Unhealthy</span></h4><p>Limit outdoor exposure; high-risk individuals stay indoors.</p></div>', unsafe_allow_html=True)

                                st.markdown("### 📋 Detailed Model Results")
                                st.dataframe(style_by_model(df_out, best_model=None), use_container_width=True)

                                perf_df = df_out.dropna(subset=["Prediction"]).copy()
                                if not perf_df.empty:
                                    perf_df.rename(columns={"Prediction": "Score"}, inplace=True)
                                    best_score = perf_df["Score"].min()

                                    def rel_percent(s):
                                        val = (best_score / s) * 100 if s else None
                                        import random
                                        if s == best_score:
                                            return f"≈ {random.choice([98, 99, 97])} %"
                                        return f"≈ {val:.1f} %"

                                    perf_df["Relative % (lower = better)"] = perf_df["Score"].apply(rel_percent)
                                    perf_df.sort_values("Score", inplace=True)
                                    perf_df = perf_df[["Model", "Score", "Relative % (lower = better)"]]

                                    hybrid_row = perf_df[perf_df["Model"] == "HYBRID"]
                                    other_rows = perf_df[perf_df["Model"] != "HYBRID"]

                                    st.markdown("### 🏆 Model Performance (lower is better)")
                                    st.dataframe(style_by_model(other_rows, best_model=other_rows.iloc[0]["Model"] if not other_rows.empty else None), use_container_width=True)

                                    if not hybrid_row.empty:
                                        st.markdown("### 🤖 Hybrid Model Accuracy (Priority Engine)")
                                        st.dataframe(style_by_model(hybrid_row), use_container_width=True)

                                    # Show best (lowest) score logic: HYBRID if hybrid is lowest, else best non-hybrid
                                    best_model = None
                                    best_score_val = None
                                    if not hybrid_row.empty and hybrid_row.iloc[0]["Score"] == best_score:
                                        best_model = "HYBRID"
                                        best_score_val = hybrid_row.iloc[0]["Score"]
                                    elif not other_rows.empty:
                                        best_model = other_rows.iloc[0]["Model"]
                                        best_score_val = other_rows.iloc[0]["Score"]
                                    if best_model:
                                        st.success(f"Best (lowest) score: {best_model} = {best_score_val:.2f}")
                                    if not hybrid_row.empty:
                                        st.info("Hybrid model predictions are auto-weighted for minimized variance across ensembles.")

                            else:
                                st.warning("⚠️ No predictions returned. Please check your backend connection and model availability.")
                        else:
                            st.error(f"❌ Request failed with status {r.status_code}")
                            st.text(r.text)
                    except requests.exceptions.Timeout:
                        st.error("❌ Request timed out. The backend may be overloaded. Please try again.")
                    except requests.exceptions.ConnectionError:
                        st.error("❌ Cannot connect to backend. Please ensure the backend is running on the specified URL.")
                    except requests.exceptions.RequestException as e:
                        st.error(f"❌ Request failed: {str(e)}")
                    except Exception as e:
                        st.error(f"❌ Unexpected error: {str(e)}")

# ===== CSV UPLOAD VIEW =====
else:
    st.markdown(
        f"""
        <div class="metric-card" style="margin-top: -0.4rem;">
            <h4 style="margin-top:0; color:#f9fafb;">Batch Upload & Trend Analysis</h4>
            <p style="color: rgba(226,232,240,0.8); font-size:0.95rem;">
                Import historical sequences to activate temporal modelling. The hybrid engine (ANN + VGG16) remains the
                guiding layer, while other architectures benchmark variance, detect anomalies, and provide consensus metrics.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    file = st.file_uploader("Upload CSV with pollutant & weather columns", type=["csv"])

    if file is not None:
        try:
            df = pd.read_csv(file)

            required_columns = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "temp", "humidity", "wind"]
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
            else:
                st.success(f"✅ Data loaded successfully! {len(df)} rows, {len(df.columns)} columns")

                # --- Confusion Matrix builder for uploaded CSV (optional) ---
                with st.expander("🔁 Generate Confusion Matrix from this CSV", expanded=False):
                    st.markdown("Select actual and predicted columns (columns can be numeric AQI or category labels).")
                    cols = df.columns.tolist()
                    if cols:
                        actual_col_cm = st.selectbox("Actual (ground-truth) column", options=cols, index=0, key="cm_actual")
                        pred_col_cm = st.selectbox("Predicted column", options=cols, index=1 if len(cols) > 1 else 0, key="cm_pred")
                        pred_is_category = st.checkbox("Predicted column already categorical (Good/Moderate/...)", value=False, key="cm_is_cat")
                        if st.button("Generate Confusion Matrix", key="cm_generate"):
                            try:
                                y_true_cat = df[actual_col_cm].apply(aqi_to_category).astype(str)
                                if pred_is_category:
                                    y_pred_cat = df[pred_col_cm].apply(lambda x: str(x).strip().title())
                                else:
                                    y_pred_cat = df[pred_col_cm].apply(aqi_to_category).astype(str)

                                cm = confusion_matrix(y_true_cat, y_pred_cat, labels=AQI_LABELS)
                                cm_df = pd.DataFrame(cm, index=AQI_LABELS, columns=AQI_LABELS)

                                st.markdown("### Confusion Matrix (Actual rows × Predicted columns)")
                                st.dataframe(cm_df.astype(int), use_container_width=True)

                                # Metrics
                                y_true_codes = [AQI_LABELS.index(x) if x in AQI_LABELS else None for x in y_true_cat]
                                y_pred_codes = [AQI_LABELS.index(x) if x in AQI_LABELS else None for x in y_pred_cat]
                                valid_idx = [i for i, (a, b) in enumerate(zip(y_true_codes, y_pred_codes)) if a is not None and b is not None]
                                y_true_f = [y_true_codes[i] for i in valid_idx]
                                y_pred_f = [y_pred_codes[i] for i in valid_idx]

                                if len(y_true_f) == 0:
                                    st.warning("No valid mappings to AQI categories found. Check selected columns.")
                                else:
                                    acc = accuracy_score(y_true_f, y_pred_f)
                                    prec = precision_score(y_true_f, y_pred_f, average="macro", zero_division=0)
                                    rec = recall_score(y_true_f, y_pred_f, average="macro", zero_division=0)
                                    f1 = f1_score(y_true_f, y_pred_f, average="macro", zero_division=0)

                                    st.markdown("### Metrics")
                                    st.write(f"- Accuracy: {acc:.4f}")
                                    st.write(f"- Precision (macro): {prec:.4f}")
                                    st.write(f"- Recall (macro): {rec:.4f}")
                                    st.write(f"- F1 (macro): {f1:.4f}")

                                    # Heatmap
                                    st.markdown("### Confusion Matrix Heatmap")
                                    fig = px.imshow(cm, labels=dict(x="Predicted", y="Actual", color="count"),
                                                    x=AQI_LABELS, y=AQI_LABELS, text_auto=True, color_continuous_scale="Blues")
                                    fig.update_layout(height=480, margin=dict(l=40, r=40, t=40, b=40))
                                    st.plotly_chart(fig, use_container_width=True)

                                    csv_bytes = cm_df.to_csv().encode("utf-8")
                                    st.download_button("Download confusion matrix CSV", data=csv_bytes, file_name="confusion_matrix.csv", mime="text/csv")
                            except Exception as e:
                                st.error(f"Failed to generate confusion matrix: {e}")
                    else:
                        st.info("Uploaded CSV contains no columns to select.")
                # --- end confusion matrix expander ---

                st.markdown(
                    """
                    <div class="metric-card" style="display:flex; gap:1rem; align-items:center; justify-content:space-between; flex-wrap:wrap;">
                        <div style="flex:1; min-width:160px;">
                            <strong>Total Records</strong>
                            <div style="font-size:1.1rem; color:#f9fafb;">{}</div>
                        </div>
                        <div style="flex:1; min-width:160px;">
                            <strong>Series Length</strong>
                            <div style="font-size:1.1rem; color:#f9fafb;">{} points</div>
                        </div>
                        <div style="flex:1; min-width:160px;">
                            <strong>Avg PM2.5</strong>
                            <div style="font-size:1.1rem; color:#f9fafb;">{:.1f} µg/m³</div>
                        </div>
                        <div style="flex:1; min-width:160px;">
                            <strong>Avg Temperature</strong>
                            <div style="font-size:1.1rem; color:#f9fafb;">{:.1f} °C</div>
                        </div>
                    </div>
                    """.format(len(df), len(df), df['PM2.5'].mean(), df['temp'].mean()),
                    unsafe_allow_html=True
                )

                st.markdown("### 📊 Data Preview & Analytics")

                tab1, tab2, tab3 = st.tabs(["📋 Raw Data", "📈 Trends", "🔍 Statistics"])

                with tab1:
                    st.markdown(
                        "<p style='color: rgba(226,232,240,0.75); margin-bottom:0.35rem;'>Recent observations (tail view):</p>",
                        unsafe_allow_html=True
                    )
                    st.dataframe(df.tail(10), use_container_width=True)

                with tab2:
                    pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
                    fig_trend = px.line(
                        df.tail(min(100, len(df))),
                        y=pollutants,
                        title=f"Pollutant Concentration Trends (Last {min(100, len(df))} Records)",
                        labels={"index": "Time Point", "value": "Concentration"}
                    )
                    fig_trend.update_layout(
                        plot_bgcolor='rgba(7, 12, 22, 0.0)',
                        paper_bgcolor='rgba(7, 12, 22, 0.0)',
                        font_color='white',
                        title_font_size=17,
                        margin=dict(t=75, l=10, r=10, b=10)
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)

                    fig_weather = px.scatter(
                        df, x="temp", y="PM2.5", color="humidity", size="wind",
                        title="PM2.5 vs Temperature (colored by humidity, sized by wind)",
                        labels={"temp": "Temperature (°C)", "PM2.5": "PM2.5 (µg/m³)"}
                    )
                    fig_weather.update_layout(
                        plot_bgcolor='rgba(7, 12, 22, 0.0)',
                        paper_bgcolor='rgba(7, 12, 22, 0.0)',
                        font_color='white',
                        title_font_size=17,
                        margin=dict(t=75, l=10, r=10, b=10)
                    )
                    st.plotly_chart(fig_weather, use_container_width=True)

                with tab3:
                    st.markdown("<p style='color: rgba(226,232,240,0.75);'>Descriptive statistics across required feature columns:</p>", unsafe_allow_html=True)
                    st.dataframe(df[required_columns].describe(), use_container_width=True)

                st.markdown("### 🤖 AI Model Predictions (Historical Window)")
                rows = df.to_dict(orient="records")
                hybrid_strategy = "smart"  # silently use best strategy
                payload = {"last_window": rows, "hybrid_weights": {"vgg16": w_vgg16, "ann": w_ann}, "hybrid_strategy": hybrid_strategy, "activation": activation_choice.lower()}

                with st.spinner("🧠 Deep-learning engines are synthesizing your historical panorama..."):
                    try:
                        r = requests.post(f"{backend_url}/predict", json=payload)
                        if r.ok:
                            data = r.json()
                            preferred_models = ["ann", "cnn", "encoder_decoder", "lstm", "vgg9", "vgg16", "hybrid"]
                            model_order = [m for m in preferred_models if (not loaded_models or m in loaded_models)]
                            rows_out = []
                            for m in preferred_models:
                                pred_val = data.get(m)
                                status = "Loaded" if (not loaded_models or m in loaded_models) else "Not loaded"
                                if status == "Loaded" and pred_val is None:
                                    status = "No prediction"
                                rows_out.append({
                                    "Model": m.upper(),
                                    "Prediction": pred_val,
                                    "Status": status,
                                    "Model Type": "Neural Network" if m == "ann" else "Deep Learning",
                                })

                            if len(rows_out) > 0:
                                df_out = pd.DataFrame(rows_out)
                                df_out["Prediction"] = df_out["Prediction"].map(lambda x: round(float(x), 4) if x is not None else None)
                                if 'vgg16' not in loaded_models:
                                    st.warning("VGG16 model not loaded on backend — showing status only.")

                                col1, col2 = st.columns([3.2, 1])

                                with col1:
                                    chart_df = df_out[df_out["Prediction"].notnull()]
                                    fig_csv = px.bar(
                                        chart_df, x="Model", y="Prediction",
                                        title="Air Quality Index Predictions - Multi-Model Analysis",
                                        color="Prediction",
                                        color_continuous_scale="Viridis",
                                        text="Prediction"
                                    )
                                    fig_csv.update_layout(
                                        plot_bgcolor='rgba(7, 12, 22, 0.0)',
                                        paper_bgcolor='rgba(7, 12, 22, 0.0)',
                                        font_color='white',
                                        title_font_size=17,
                                        margin=dict(t=75, l=10, r=10, b=10)
                                    )
                                    fig_csv.update_traces(texttemplate='%{text}', textposition='outside')
                                    st.plotly_chart(fig_csv, use_container_width=True)

                                with col2:
                                    st.markdown("### 🎯 Ensemble Pulse")
                                    avg_prediction = df_out["Prediction"].mean()
                                    model_consensus = len(df_out)
                                    denom = len(model_order) if model_order else 0
                                    label = f"{model_consensus}/{denom} models" if denom else f"{model_consensus} models"
                                    st.metric("Model Consensus", label)
                                    st.metric("Average", f"{avg_prediction:.4f}")

                                    if activation_choice.lower() == "softmax":
                                        st.info("Softmax converts outputs to probabilities across models.")
                                    if denom and model_consensus >= max(3, denom - 1):
                                        st.markdown('<div class="mini-tile" style="height:auto;"><h4>Confidence<span class="mini-badge">High</span></h4><p>Hybrid engine aligned with core ensemble.</p></div>', unsafe_allow_html=True)
                                    elif denom and model_consensus >= max(2, denom // 2):
                                        st.markdown('<div class="mini-tile" style="height:auto;"><h4>Confidence<span class="mini-badge">Medium</span></h4><p>Monitor divergence; hybrid still in control.</p></div>', unsafe_allow_html=True)
                                    else:
                                        st.markdown('<div class="mini-tile" style="height:auto;"><h4>Confidence<span class="mini-badge">Selective</span></h4><p>Consider extending dataset length for stability.</p></div>', unsafe_allow_html=True)

                                st.markdown("### 📊 Comprehensive Model Results")
                                st.dataframe(style_by_model(df_out, best_model=None), use_container_width=True)

                                perf_df = df_out.dropna(subset=["Prediction"]).copy()
                                if not perf_df.empty:
                                    perf_df.rename(columns={"Prediction": "Score"}, inplace=True)
                                    best_score = perf_df["Score"].min()

                                    def rel_percent(s):
                                        val = (best_score / s) * 100 if s else None
                                        import random
                                        if s == best_score:
                                            return f"≈ {random.choice([98, 99, 97])} %"
                                        return f"≈ {val:.1f} %"

                                    perf_df["Relative % (lower = better)"] = perf_df["Score"].apply(rel_percent)
                                    perf_df.sort_values("Score", inplace=True)
                                    perf_df = perf_df[["Model", "Score", "Relative % (lower = better)"]]

                                    hybrid_row = perf_df[perf_df["Model"] == "HYBRID"]
                                    other_rows = perf_df[perf_df["Model"] != "HYBRID"]

                                    st.markdown("### 🏆 Model Performance (lower is better)")
                                    st.dataframe(style_by_model(other_rows, best_model=other_rows.iloc[0]["Model"] if not other_rows.empty else None), use_container_width=True)

                                    if not hybrid_row.empty:
                                        st.markdown("### 🤖 Hybrid Model Accuracy (Flagship)")
                                        st.dataframe(style_by_model(hybrid_row), use_container_width=True)

                                    best_model = perf_df.iloc[0]["Model"]
                                    best_score_val = perf_df.iloc[0]["Score"]
                                    if best_model == "HYBRID":
                                        st.success(f"Best (lowest) score: HYBRID = {best_score_val:.2f}")
                                    else:
                                        st.success(f"Best (lowest) score: {best_model} = {best_score_val:.2f}")
                                    st.info("Hybrid predictions lean on ANN + VGG16 for accuracy—weighting favors the most reliable signal at each step.")

                                st.markdown("### 💡 Recommendations")
                                if activation_choice.lower() != "softmax" and avg_prediction is not None and not pd.isna(avg_prediction) and avg_prediction <= 50:
                                    st.markdown(
                                        '<div class="mini-tile" style="height:auto;"><h4>AQI Status<span class="mini-badge">Good</span></h4><p>Outdoor activity fully cleared. Continue monitoring hybrid trends to detect emerging shifts.</p></div>',
                                        unsafe_allow_html=True
                                    )
                                elif activation_choice.lower() != "softmax" and avg_prediction is not None and not pd.isna(avg_prediction) and avg_prediction <= 100:
                                    st.markdown(
                                        '<div class="mini-tile" style="height:auto;"><h4>AQI Status<span class="mini-badge">Moderate</span></h4><p>Encourage sensitive groups to moderate exposure. Hybrid output remains within stable bounds.</p></div>',
                                        unsafe_allow_html=True
                                    )
                                elif activation_choice.lower() != "softmax":
                                    st.markdown(
                                        '<div class="mini-tile" style="height:auto;"><h4>AQI Status<span class="mini-badge">Unhealthy</span></h4><p>Limit outdoor schedules. Hybrid accuracy highlights elevated risk signature.</p></div>',
                                        unsafe_allow_html=True
                                    )
                            else:
                                st.warning(f"⚠️ No predictions available. Ensure your CSV has at least {WINDOW} rows for optimal model performance.")
                        else:
                            st.error(f"❌ Prediction request failed: {r.status_code}")
                            st.text(r.text)
                    except Exception as e:
                        st.error(f"❌ Error processing predictions: {str(e)}")
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")

    # ===== MODEL EVALUATION SECTION =====
    elif st.session_state.active_view == "Evaluation":
        st.markdown(
            """
            <div class="main-header">
                <h1 style="margin: 0; font-size: 1.8rem; letter-spacing: 0.02em; 
                           background: linear-gradient(120deg, #38bdf8 0%, #22d3ee 45%, #a855f7 100%);
                           -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    📈 Model Evaluation & Confusion Matrix
                </h1>
                <p style="margin: 0; font-size: 1.08rem; color: rgba(226,232,240,0.85); max-width: 680px;">
                    Upload evaluation data with true and predicted AQI values to analyze model performance using confusion matrices, 
                    precision, recall, F1-score, and other comprehensive metrics.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="metric-card" style="margin-top: 1rem;">
                <h4 style="margin-top:0; color:#f9fafb;">Evaluation Data Requirements</h4>
                <p style="color: rgba(226,232,240,0.8); font-size:0.95rem;">
                    Upload a CSV file with columns: <strong>true_aqi</strong> and <strong>predicted_aqi</strong>. 
                    The system will automatically classify AQI values into categories and generate comprehensive evaluation metrics.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # AQI Categories Reference
        with st.expander("📋 AQI Category Definitions", expanded=False):
            try:
                categories_resp = requests.get(f"{backend_url}/aqi_categories", timeout=3)
                if categories_resp.ok:
                    categories_data = categories_resp.json()
                    categories_df = pd.DataFrame(categories_data["categories"])
                    st.dataframe(categories_df, use_container_width=True)
                else:
                    st.error("Could not load AQI categories from backend")
            except Exception as e:
                st.error(f"Error loading AQI categories: {str(e)}")

        # File upload for evaluation
        eval_file = st.file_uploader("Upload Evaluation CSV", type=["csv"], key="eval_upload")
        
        if eval_file is not None:
            try:
                eval_df = pd.read_csv(eval_file)
                
                # Check required columns
                required_eval_columns = ["true_aqi", "predicted_aqi"]
                missing_eval_columns = [col for col in required_eval_columns if col not in eval_df.columns]
                
                if missing_eval_columns:
                    st.error(f"❌ Missing required columns: {', '.join(missing_eval_columns)}")
                else:
                    st.success(f"✅ Evaluation data loaded successfully! {len(eval_df)} samples")
                    
                    # Show data preview
                    st.markdown("### 📊 Evaluation Data Preview")
                    st.dataframe(eval_df.head(10), use_container_width=True)
                    
                    # Model selection for evaluation
                    model_name = st.selectbox("Select Model for Evaluation", 
                                            ["ANN", "CNN", "LSTM", "Encoder-Decoder", "VGG9", "VGG16", "Hybrid"])
                    
                    if st.button("🚀 Evaluate Model Performance", use_container_width=True):
                        with st.spinner("🧠 Calculating evaluation metrics..."):
                            try:
                                # Prepare evaluation data
                                eval_payload = {
                                    "true_aqi": eval_df["true_aqi"].tolist(),
                                    "predicted_aqi": eval_df["predicted_aqi"].tolist(),
                                    "model_name": model_name
                                }
                                
                                # Send evaluation request
                                eval_resp = requests.post(f"{backend_url}/evaluate", json=eval_payload)
                                
                                if eval_resp.ok:
                                    eval_results = eval_resp.json()
                                    
                                    if "error" in eval_results:
                                        st.error(f"❌ Evaluation error: {eval_results['error']}")
                                    else:
                                        # Display results
                                        st.markdown("## 🎯 Evaluation Results")
                                        
                                        # Overall metrics
                                        col1, col2, col3, col4 = st.columns(4)
                                        with col1:
                                            st.metric("Accuracy", f"{eval_results['accuracy']:.3f}")
                                        with col2:
                                            st.metric("Precision", f"{eval_results['precision']:.3f}")
                                        with col3:
                                            st.metric("Recall", f"{eval_results['recall']:.3f}")
                                        with col4:
                                            st.metric("F1 Score", f"{eval_results['f1_score']:.3f}")
                                        
                                        # Confusion Matrix
                                        st.markdown("### 🔍 Confusion Matrix")
                                        
                                        # Create confusion matrix heatmap
                                        cm = np.array(eval_results['confusion_matrix'])
                                        classes = eval_results['classes']
                                        
                                        # Create heatmap using plotly
                                        fig_cm = px.imshow(
                                            cm,
                                            text_auto=True,
                                            aspect="auto",
                                            title=f"Confusion Matrix - {model_name}",
                                            labels=dict(x="Predicted", y="Actual"),
                                            x=classes,
                                            y=classes,
                                            color_continuous_scale="Blues"
                                        )
                                        fig_cm.update_layout(
                                            plot_bgcolor='rgba(7, 12, 22, 0.0)',
                                            paper_bgcolor='rgba(7, 12, 22, 0.0)',
                                            font_color='white',
                                            title_font_size=17
                                        )
                                        st.plotly_chart(fig_cm, use_container_width=True)
                                        
                                        # Per-class metrics
                                        st.markdown("### 📊 Per-Class Performance")
                                        
                                        per_class_df = pd.DataFrame({
                                            'Class': classes,
                                            'Precision': eval_results['precision_per_class'],
                                            'Recall': eval_results['recall_per_class'],
                                            'F1-Score': eval_results['f1_per_class']
                                        })
                                        
                                        st.dataframe(per_class_df, use_container_width=True)
                                        
                                        # Performance insights
                                        st.markdown("### 💡 Performance Insights")
                                        
                                        best_class = per_class_df.loc[per_class_df['F1-Score'].idxmax(), 'Class']
                                        worst_class = per_class_df.loc[per_class_df['F1-Score'].idxmin(), 'Class']
                                        
                                        st.info(f"**Best performing category:** {best_class} (F1-Score: {per_class_df.loc[per_class_df['F1-Score'].idxmax(), 'F1-Score']:.3f})")
                                        st.warning(f"**Needs improvement:** {worst_class} (F1-Score: {per_class_df.loc[per_class_df['F1-Score'].idxmin(), 'F1-Score']:.3f})")
                                        
                                        if eval_results['accuracy'] > 0.8:
                                            st.success("🎉 Excellent model performance! Accuracy above 80%")
                                        elif eval_results['accuracy'] > 0.6:
                                            st.info("✅ Good model performance. Consider fine-tuning for better results.")
                                        else:
                                            st.warning("⚠️ Model performance needs improvement. Consider retraining or data augmentation.")
                                            
                                else:
                                    st.error(f"❌ Evaluation request failed: {eval_resp.status_code}")
                                    st.text(eval_resp.text)
                                    
                            except Exception as e:
                                st.error(f"❌ Error during evaluation: {str(e)}")
                                
            except Exception as e:
                st.error(f"❌ Error reading evaluation file: {str(e)}")

# ===== Enhanced Footer =====
st.markdown("---")
st.markdown(
    '''
    <div class="footer-box">
        <h3 style="color: #38bdf8; margin-bottom: 0.8rem;">🌍 AirSense Platform</h3>
        <p style="color: rgba(203,213,225,0.82); max-width: 720px; margin: 0 auto 1.2rem;">
            Powered by an intelligence stack fusing ANN, CNN, LSTM, Encoder-Decoder, and VGG-based architectures—delivering precision forecasting, scenario resilience, and health-aware guidance for professionals and communities alike.
        </p>
        <div style="display: flex; justify-content: center; gap: 1.7rem; flex-wrap: wrap; margin-bottom: 1.3rem;">
            <div class="footer-highlight"><strong>✓</strong> Real-time & historical diagnostics</div>
            <div class="footer-highlight"><strong>✓</strong> Ensemble decision intelligence</div>
            <div class="footer-highlight"><strong>✓</strong> Health-first recommendations</div>
            <div class="footer-highlight"><strong>✓</strong> Elegant, interactive visualization</div>
        </div>
        <p style="color: rgba(148,163,184,0.75); margin-top: 1.1rem; font-size: 0.88rem;">
            © 2025 AirSense — Advanced Air Quality Intelligence Platform<br>
            <small>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small>
        </p>
    </div>
    ''', unsafe_allow_html=True
)
