import streamlit as st
import requests
import pandas as pd
import os, sys
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Ensure project root is on PYTHONPATH for `src` imports when running from `frontend/`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import WINDOW

st.set_page_config(
    page_title="AirSense — AI-Powered Air Quality Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===== Modern Professional Theme =====
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%);
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid #334155;
        margin-bottom: 1px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #2d3748 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #4a5568;
        margin: 0.5rem 0;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    }
    
    /* Flexible height and consistent styling for all information cards */
    .info-card {
        background: linear-gradient(135deg, #065f46 0%, #047857 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #10b981;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(16, 185, 129, 0.1);
        min-height: 180px;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .i-card {
        background: linear-gradient(135deg, #065f46 0%, #047857 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #10b981;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(16, 185, 129, 0.1);
       min-height: 420px;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .info-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.2);
    }
    
    .warning-card {
        background: linear-gradient(135deg, #92400e 0%, #d97706 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #f59e0b;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(245, 158, 11, 0.1);
        min-height: 180px;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .w-card {
        background: linear-gradient(135deg, #92400e 0%, #d97706 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #f59e0b;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(245, 158, 11, 0.1);
        min-height: 420px;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .warning-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(245, 158, 11, 0.2);
    }
    
    .danger-card {
        background: linear-gradient(135deg, #991b1b 0%, #dc2626 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #ef4444;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(239, 68, 68, 0.1);
        min-height: 180px;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .danger-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(239, 68, 68, 0.2);
    }
    
    .pollutant-info {
        background: linear-gradient(135deg, #1e40af 0%, #3730a3 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 1px solid #3b82f6;
        min-height: 180px;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.1);
    }
    .p-info {
        background: linear-gradient(135deg, #1e40af 0%, #3730a3 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 1px solid #3b82f6;
        min-height: 420px;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.1);
    }
    
    .pollutant-info:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.2);
    }
    
    /* Enhanced card content styling with better spacing */
    .info-card h4, .warning-card h4, .danger-card h4, .pollutant-info h4 {
        margin-top: 0;
        margin-bottom: 1rem;
        font-size: 1.1rem;
        font-weight: 600;
        color: #ffffff;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
    }
    
    .info-card strong, .warning-card strong, .danger-card strong, .pollutant-info strong {
        color: #ffffff;
        font-weight: 600;
    }
    
    .info-card p, .warning-card p, .danger-card p, .pollutant-info p {
        line-height: 1.6;
        margin-bottom: 0.5rem;
    }
    
    .sidebar .stSelectbox > div > div {
        background-color: #2d3748;
        border: 1px solid #4a5568;
        border-radius: 8px;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s ease;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(59, 130, 246, 0.4);
    }
    
    .prediction-table {
        background: #1e293b;
        border-radius: 12px;
        border: 1px solid #334155;
        overflow: hidden;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #f1f5f9;
        font-weight: 600;
    }
    
    .stMetric {
        background: linear-gradient(135deg, #1e293b 0%, #2d3748 100%);
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #4a5568;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ===== Enhanced Sidebar =====
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; padding: 1rem;">
            <h2 style="color: #3b82f6; margin-bottom: 0.5rem;">🌍 AirSense</h2>
            <p style="color: #94a3b8; font-size: 0.9rem;">AI-Powered Air Quality Intelligence</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    
    st.markdown("### ⚙️ Configuration")
    backend_url = st.text_input("Backend URL", value="http://localhost:8000")
    
    st.markdown("### 📊 Model Information")
    st.markdown(
        f"""
        <div class="info-card">
            <strong>Window Size:</strong> {WINDOW} data points<br>
            <strong>Models Available:</strong><br>
            • ANN (Artificial Neural Network)<br>
            • CNN (Convolutional Neural Network)<br>
            • LSTM (Long Short-Term Memory)<br>
            • Encoder-Decoder<br>
            • VGG9 (Visual Geometry Group)
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("### 🎯 Quick Tips")
    st.markdown(
        """
        <div class="metric-card">
            <strong>For Best Results:</strong><br>
            • Upload CSV with ≥{} rows<br>
            • Ensure all pollutant columns<br>
            • Check data quality<br>
            • Monitor prediction trends
        </div>
        """.format(WINDOW),
        unsafe_allow_html=True
    )

# ===== Main Header =====
st.markdown(
    """
    <div class="main-header">
        <h1 style="margin: 0; font-size: 1.5rem; background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            🌍 AirSense Dashboard
        </h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; color: #94a3b8;">
            Advanced AI-powered air quality prediction and analysis platform
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ===== Air Quality Information Cards =====
st.markdown("## 📚 Understanding Air Quality")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="p-info">
            <h4>🔴 Primary Pollutants</h4>
            <p><strong>PM2.5:</strong> Fine particles ≤2.5μm</p>
            <p><strong>PM10:</strong> Particles ≤10μm</p>
            <p><strong>NO₂:</strong> Nitrogen dioxide</p>
            <p><strong>SO₂:</strong> Sulfur dioxide</p>
            <p><strong>CO:</strong> Carbon monoxide</p>
            <p><strong>O₃:</strong> Ground-level ozone</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="i-card">
            <h4>🌡️ Environmental Factors</h4>
            <p><strong>Temperature:</strong> Affects pollutant formation</p>
            <p><strong>Humidity:</strong> Influences particle behavior</p>
            <p><strong>Wind Speed:</strong> Disperses pollutants</p>
            <p><strong>Atmospheric Pressure:</strong> Affects air movement</p>
            <p><strong>Solar Radiation:</strong> Drives photochemical reactions</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div class="w-card">
            <h4>⚠️ Health Impact Levels</h4>
            <p><strong>Good (0-50):</strong> Minimal impact</p>
            <p><strong>Moderate (51-100):</strong> Sensitive groups</p>
            <p><strong>Unhealthy for Sensitive (101-150):</strong> Everyone affected</p>
            <p><strong>Unhealthy (151-200):</strong> Everyone affected</p>
            <p><strong>Very Unhealthy (201-300):</strong> Emergency conditions</p>
            <p><strong>Hazardous (301+):</strong> Emergency</p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# ===== Enhanced Manual Input Section =====
with st.expander("✏️ Manual Air Quality Input", expanded=False):
    st.markdown(
        """
        <div class="info-card">
            <h4>📝 Single Reading Analysis</h4>
            <p>Enter current air quality measurements to get instant AI predictions. 
            The system will create a {}-step window internally for comprehensive model analysis.</p>
        </div>
        """.format(WINDOW),
        unsafe_allow_html=True
    )
    
    with st.form("manual_form"):
        st.markdown("### 🌬️ Pollutant Measurements")
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
        features = {
            "PM2.5": pm25, "PM10": pm10, "NO2": no2, "SO2": so2, 
            "CO": co, "O3": o3, "temp": temp, "humidity": humidity, "wind": wind
        }
        rows = [features for _ in range(WINDOW)]
        payload = {"features": features, "last_window": rows}
        
        with st.spinner("🤖 AI models are analyzing your data..."):
            try:
                r = requests.post(f"{backend_url}/predict", json=payload)
                if r.ok:
                    data = r.json()
                    model_order = ["ann", "cnn", "encoder_decoder", "lstm", "vgg9"]
                    rows_out = [{"Model": m.upper(), "Prediction": data.get(m), "Confidence": "High" if m in ["ann", "lstm"] else "Medium"} 
                               for m in model_order if m in data]
                    
                    if len(rows_out) > 0:
                        df_out = pd.DataFrame(rows_out)
                        df_out["Prediction"] = df_out["Prediction"].map(lambda x: round(float(x), 2) if x is not None else None)
                        
                        st.markdown("## 🔮 AI Model Predictions")
                        
                        # Enhanced prediction display
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # Create enhanced bar chart
                            fig_manual = px.bar(
                                df_out, x="Model", y="Prediction", 
                                title="Air Quality Index Predictions by AI Model",
                                color="Prediction",
                                color_continuous_scale="RdYlBu_r",
                                text="Prediction"
                            )
                            fig_manual.update_layout(
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font_color='white',
                                title_font_size=16
                            )
                            fig_manual.update_traces(texttemplate='%{text}', textposition='outside')
                            st.plotly_chart(fig_manual, use_container_width=True)
                        
                        with col2:
                            st.markdown("### 📊 Prediction Summary")
                            avg_prediction = df_out["Prediction"].mean()
                            max_prediction = df_out["Prediction"].max()
                            min_prediction = df_out["Prediction"].min()
                            
                            st.metric("Average AQI", f"{avg_prediction:.1f}")
                            st.metric("Highest Prediction", f"{max_prediction:.1f}")
                            st.metric("Lowest Prediction", f"{min_prediction:.1f}")
                            
                            # Health recommendation
                            if avg_prediction <= 50:
                                st.markdown('<div class="info-card"><strong>Status:</strong> Good Air Quality</div>', unsafe_allow_html=True)
                            elif avg_prediction <= 100:
                                st.markdown('<div class="warning-card"><strong>Status:</strong> Moderate - Sensitive groups should limit outdoor activities</div>', unsafe_allow_html=True)
                            else:
                                st.markdown('<div class="danger-card"><strong>Status:</strong> Unhealthy - Limit outdoor activities</div>', unsafe_allow_html=True)
                        
                        # Detailed results table
                        st.markdown("### 📋 Detailed Model Results")
                        st.dataframe(df_out, use_container_width=True)
                        
                    else:
                        st.warning("⚠️ No predictions returned. Please check your backend connection and model availability.")
                else:
                    st.error(f"❌ Request failed with status {r.status_code}")
                    st.text(r.text)
            except Exception as e:
                st.error(f"❌ Connection error: {str(e)}")

st.markdown("---")

# ===== Enhanced CSV Upload Section =====
with st.expander("📂 Upload Historical Data (CSV)", expanded=True):
    st.markdown(
        """
        <div class="info-card">
            <h4>📈 Batch Analysis & Trend Prediction</h4>
            <p>Upload historical air quality data for comprehensive analysis. 
            CSV must contain columns: <code>PM2.5, PM10, NO2, SO2, CO, O3, temp, humidity, wind</code><br>
            <strong>Minimum {0} rows required</strong> for advanced models (CNN, LSTM, Encoder-Decoder, VGG9)</p>
        </div>
        """.format(WINDOW),
        unsafe_allow_html=True
    )
    
    file = st.file_uploader("Choose CSV file", type=["csv"], help="Upload your air quality measurement data")

    if file is not None:
        try:
            df = pd.read_csv(file)
            
            # Data validation
            required_columns = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "temp", "humidity", "wind"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
            else:
                st.success(f"✅ Data loaded successfully! {len(df)} rows, {len(df.columns)} columns")
                
                # Data overview
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Records", len(df))
                with col2:
                    st.metric("Date Range", f"{len(df)} points")
                with col3:
                    st.metric("Avg PM2.5", f"{df['PM2.5'].mean():.1f} µg/m³")
                with col4:
                    st.metric("Avg Temperature", f"{df['temp'].mean():.1f}°C")
                
                # Enhanced data preview
                st.markdown("### 📊 Data Preview & Analysis")
                
                tab1, tab2, tab3 = st.tabs(["📋 Raw Data", "📈 Trends", "🔍 Statistics"])
                
                with tab1:
                    st.dataframe(df.tail(10), use_container_width=True)
                
                with tab2:
                    # Pollutant trends
                    pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
                    fig_trend = px.line(
                        df.tail(min(100, len(df))), 
                        y=pollutants, 
                        title=f"Pollutant Concentration Trends (Last {min(100, len(df))} Records)",
                        labels={"index": "Time Point", "value": "Concentration"}
                    )
                    fig_trend.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='white'
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)
                    
                    # Weather correlation
                    fig_weather = px.scatter(
                        df, x="temp", y="PM2.5", color="humidity", size="wind",
                        title="PM2.5 vs Temperature (colored by humidity, sized by wind)",
                        labels={"temp": "Temperature (°C)", "PM2.5": "PM2.5 (µg/m³)"}
                    )
                    fig_weather.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='white'
                    )
                    st.plotly_chart(fig_weather, use_container_width=True)
                
                with tab3:
                    st.markdown("### 📈 Statistical Summary")
                    st.dataframe(df[required_columns].describe(), use_container_width=True)

                # AI Predictions
                st.markdown("### 🤖 AI Model Predictions")
                rows = df.to_dict(orient="records")
                payload = {"last_window": rows}
                
                with st.spinner("🧠 Advanced AI models are processing your data..."):
                    try:
                        r = requests.post(f"{backend_url}/predict", json=payload)
                        if r.ok:
                            data = r.json()
                            model_order = ["ann", "cnn", "encoder_decoder", "lstm", "vgg9"]
                            rows_out = [{"Model": m.upper(), "Prediction": data.get(m), "Model Type": "Neural Network" if m == "ann" else "Deep Learning"} 
                                       for m in model_order if m in data]
                            
                            if len(rows_out) > 0:
                                df_out = pd.DataFrame(rows_out)
                                df_out["Prediction"] = df_out["Prediction"].map(lambda x: round(float(x), 2) if x is not None else None)
                                
                                # Enhanced visualization
                                col1, col2 = st.columns([3, 1])
                                
                                with col1:
                                    # Create professional prediction chart
                                    fig_csv = px.bar(
                                        df_out, x="Model", y="Prediction",
                                        title="Air Quality Index Predictions - Multi-Model Analysis",
                                        color="Prediction",
                                        color_continuous_scale="Viridis",
                                        text="Prediction"
                                    )
                                    fig_csv.update_layout(
                                        plot_bgcolor='rgba(0,0,0,0)',
                                        paper_bgcolor='rgba(0,0,0,0)',
                                        font_color='white',
                                        title_font_size=16
                                    )
                                    fig_csv.update_traces(texttemplate='%{text}', textposition='outside')
                                    st.plotly_chart(fig_csv, use_container_width=True)
                                
                                with col2:
                                    st.markdown("### 🎯 Analysis Results")
                                    avg_prediction = df_out["Prediction"].mean()
                                    model_consensus = len(df_out)
                                    
                                    st.metric("Model Consensus", f"{model_consensus}/5 models")
                                    st.metric("Average AQI", f"{avg_prediction:.1f}")
                                    
                                    # Confidence indicator
                                    if model_consensus >= 4:
                                        st.markdown('<div class="info-card"><strong>Confidence:</strong> High</div>', unsafe_allow_html=True)
                                    elif model_consensus >= 2:
                                        st.markdown('<div class="warning-card"><strong>Confidence:</strong> Medium</div>', unsafe_allow_html=True)
                                    else:
                                        st.markdown('<div class="danger-card"><strong>Confidence:</strong> Low</div>', unsafe_allow_html=True)
                                
                                # Professional results table
                                st.markdown("### 📊 Comprehensive Model Results")
                                st.dataframe(df_out, use_container_width=True)
                                
                                # Recommendations
                                st.markdown("### 💡 Recommendations")
                                if avg_prediction <= 50:
                                    st.markdown(
                                        '<div class="info-card"><strong>Air Quality Status:</strong> Good<br>'
                                        '<strong>Recommendation:</strong> Ideal conditions for all outdoor activities.</div>',
                                        unsafe_allow_html=True
                                    )
                                elif avg_prediction <= 100:
                                    st.markdown(
                                        '<div class="warning-card"><strong>Air Quality Status:</strong> Moderate<br>'
                                        '<strong>Recommendation:</strong> Sensitive individuals should consider limiting prolonged outdoor exertion.</div>',
                                        unsafe_allow_html=True
                                    )
                                else:
                                    st.markdown(
                                        '<div class="danger-card"><strong>Air Quality Status:</strong> Unhealthy<br>'
                                        '<strong>Recommendation:</strong> Everyone should limit outdoor activities. Sensitive groups should avoid outdoor activities.</div>',
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

# ===== Enhanced Footer =====
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #1e293b 0%, #334155 100%); border-radius: 12px; margin-top: 2rem;">
        <h3 style="color: #3b82f6; margin-bottom: 1rem;">🌍 AirSense Platform</h3>
        <p style="color: #94a3b8; margin-bottom: 1rem;">
            Powered by advanced AI models including Neural Networks, CNNs, LSTMs, and Encoder-Decoder architectures
        </p>
        <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
            <div style="color: #10b981;"><strong>✓</strong> Real-time Analysis</div>
            <div style="color: #10b981;"><strong>✓</strong> Multi-Model Predictions</div>
            <div style="color: #10b981;"><strong>✓</strong> Health Recommendations</div>
            <div style="color: #10b981;"><strong>✓</strong> Professional Visualization</div>
        </div>
        <p style="color: #64748b; margin-top: 1rem; font-size: 0.9rem;">
            © 2025 AirSense — Advanced Air Quality Intelligence Platform
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
