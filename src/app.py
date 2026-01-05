import streamlit as st
import pandas as pd
import sys
import os
import time
import numpy as np
import plotly.express as px

# --- 1. MODEL COMPATIBILITY FIX ---
from train_ensembles import AnomalyEnsemble
sys.modules['__main__'].AnomalyEnsemble = AnomalyEnsemble

from simulator import SOCSimulator
from loaders import get_hour_data_fast

# --- 2. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI SOC Analyst Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# --- 3. STATEFUL OBJECT CACHING ---
@st.cache_resource
def load_simulator():
    st.write("🔧 Loading SOC Simulator...")
    start = time.time()
    sim_instance = SOCSimulator()
    st.write(f"✅ Simulator loaded in {time.time() - start:.2f}s")
    return sim_instance

sim = load_simulator()

# --- 4. SESSION STATE INITIALIZATION ---
if 'current_day' not in st.session_state:
    st.session_state.current_day = 1
if 'current_hour' not in st.session_state:
    st.session_state.current_hour = 0
if 'history' not in st.session_state:
    st.session_state.history = []  
if 'total_alerts_df' not in st.session_state:
    st.session_state.total_alerts_df = pd.DataFrame()

# --- 5. SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("🛡️ SOC Controls")
    st.markdown("---")
    st.info(f"**Timeline:** Day {st.session_state.current_day}")
    st.info(f"**Next Slice:** {st.session_state.current_hour:02d}:00")
    
    # Feature: Diagnostic Export
    if not st.session_state.total_alerts_df.empty:
        csv = st.session_state.total_alerts_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Export Alert Report",
            data=csv,
            file_name=f"soc_alerts_day{st.session_state.current_day}.csv",
            mime='text/csv',
            use_container_width=True
        )

    if st.button("🗑️ Reset Simulation", use_container_width=True):
        st.session_state.clear()
        st.cache_resource.clear()
        st.rerun()

# --- 5b. Total Red Team Events ---
REDTEAM_PATH = os.environ.get("REDTEAM_PATH", "redteam_log.csv")  # adjust path if needed
if os.path.exists(REDTEAM_PATH):
    red_df = pd.read_csv(REDTEAM_PATH, names=["Timestamp", "User", "Src", "Dst"], header=None)
    st.sidebar.metric("🎯 Total Red Team Events", len(red_df))
else:
    st.sidebar.info("⚠️ Red Team log not found.")



# --- 6. HEADER & METRICS ---
st.title("AI-Powered SOC Analyst Dashboard")
st.caption("Monitoring LANL Network Authentication Events for Anomalous Sequences")

m1, m2, m3 = st.columns(3)
total_anomalies = len(st.session_state.total_alerts_df)
total_hits = sum([h['red_hits'] for h in st.session_state.history]) if st.session_state.history else 0

m1.metric("Timeline", f"Day {st.session_state.current_day} | {st.session_state.current_hour:02d}:00")
m2.metric("Total Anomalies Found", total_anomalies)
m3.metric("Verified Red Team Hits", total_hits, delta="Ground Truth Matches")

# --- 7. MAIN PROCESSING BLOCK ---
button_label = f"🚀 Analyze Hour {st.session_state.current_hour:02d}:00"

if st.button(button_label, type="primary", use_container_width=True):
    with st.status(f"Scanning Telemetry...", expanded=True) as status:
        total_start = time.time()
        
        # --- Step A: Ingestion ---
        st.write("📂 Accessing Parquet Partitions...")
        ingest_start = time.time()
        raw_df = get_hour_data_fast(st.session_state.current_day, st.session_state.current_hour)
        ingest_time = time.time() - ingest_start
        st.write(f"⏱ Ingest step completed in {ingest_time:.2f}s")
        st.write(f"ℹ Rows retrieved: {len(raw_df)}")
        st.write(f"ℹ Columns: {list(raw_df.columns)}")

        # Quick check to confirm we are actually touching data
        if not raw_df.empty:
            st.write("ℹ Inspect first 3 rows to ensure data loaded:")
            st.dataframe(raw_df.head(3))

            # --- Step B: LSTM & Ensemble Inference ---
            st.write("🧠 Running Neural Behavioral Analysis...")
            inference_start = time.time()
            progress_bar = st.progress(0, text="Processing Windows...")

            # Process hour with debug prints
            hour_alerts = sim.process_hour_stream(raw_df, progress_bar=progress_bar)
            inference_time = time.time() - inference_start
            st.write(f"⏱ Neural inference completed in {inference_time:.2f}s")
            st.write(f"ℹ Alerts detected: {len(hour_alerts)}")

            # --- Step C: Verification ---
            st.write("🎯 Cross-referencing Red Team Ground Truth...")
            verify_start = time.time()
            red_hits = sim.verify_red_team(hour_alerts)
            verify_time = time.time() - verify_start
            st.write(f"⏱ Verification step completed in {verify_time:.2f}s")
            st.write(f"ℹ Verified Red Team hits: {len(red_hits)}")

            # --- Update History ---
            st.session_state.history.append({
                "Day": st.session_state.current_day,
                "Hour": st.session_state.current_hour,
                "time_label": f"D{st.session_state.current_day} H{st.session_state.current_hour}",
                "alerts": len(hour_alerts),
                "red_hits": len(red_hits),
                "avg_error": hour_alerts['lstm_error'].mean() if not hour_alerts.empty else 0
            })

            if not hour_alerts.empty:
                st.session_state.total_alerts_df = pd.concat([st.session_state.total_alerts_df, hour_alerts])

            # Immediate UI Feedback
            if not red_hits.empty:
                st.error(f"🚨 CRITICAL: {len(red_hits)} Red Team actions confirmed!")
                st.dataframe(red_hits[['Timestamp', 'User', 'Src', 'Dst', 'track']], use_container_width=True)
            else:
                st.success(f"Analysis Complete. No verified threats in this slice.")

        else:
            st.warning(f"No telemetry found for Day {st.session_state.current_day}, Hour {st.session_state.current_hour}.")

        total_time = time.time() - total_start
        status.update(label=f"SOC Sequence Complete in {total_time:.2f}s", state="complete", expanded=False)
        st.write(f"🕒 Total processing time (ingest + inference + verification): {total_time:.2f}s")

    # --- Increment Time ---
    if st.session_state.current_hour < 23:
        st.session_state.current_hour += 1
    else:
        st.session_state.current_hour = 0
        st.session_state.current_day += 1
    
    st.rerun()

# --- 8. VISUALIZATIONS ---
if st.session_state.history:
    st.divider()
    history_df = pd.DataFrame(st.session_state.history)
    
    tab1, tab2 = st.tabs(["📊 Threat Trends", "🌡️ Anomaly Heatmap"])
    
    with tab1:
        col_l, col_r = st.columns(2)
        with col_l:
            st.subheader("Red Team Discovery")
            st.bar_chart(history_df, x="time_label", y="red_hits", color="#ff4b4b")
        with col_r:
            st.subheader("Anomaly Volume Trend")
            st.line_chart(history_df, x="time_label", y="alerts")

    with tab2:
        if not st.session_state.total_alerts_df.empty:
            fig = px.density_heatmap(
                st.session_state.total_alerts_df, 
                x="Timestamp", 
                y="track", 
                z="lstm_error",
                title="Behavioral Anomaly Intensity",
                color_continuous_scale="Reds",
                nbinsx=50
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Heatmap will populate as anomalies are detected.")
