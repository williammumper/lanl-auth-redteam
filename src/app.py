import streamlit as st
import pandas as pd
import sys
import os
import time
import gzip
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
if 'is_running' not in st.session_state:
    st.session_state.is_running = False

# --- 5. DATA HELPERS (Red Team GZ Handling) ---
REDTEAM_GZ_PATH = "data/redteam.txt.gz" 

@st.cache_data
def get_red_team_ground_truth():
    if os.path.exists(REDTEAM_GZ_PATH):
        with gzip.open(REDTEAM_GZ_PATH, "rt") as f:
            # LANL redteam.txt usually has 4 columns: time, user, source, dest
            return pd.read_csv(f, names=["Timestamp", "User", "Src", "Dst"], header=None)
    return pd.DataFrame()

red_ground_truth = get_red_team_ground_truth()

# --- 5b. SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("🛡️ SOC Controls")
    st.markdown("---")
    st.info(f"**Timeline:** Day {st.session_state.current_day}")
    st.info(f"**Next Slice:** {st.session_state.current_hour:02d}:00")
    
    if not red_ground_truth.empty:
        st.sidebar.metric("🎯 Total Red Team (Full File)", len(red_ground_truth))
    else:
        st.sidebar.warning("⚠️ Red Team log not found at data/redteam.txt.gz")

    if not st.session_state.total_alerts_df.empty:
        csv = st.session_state.total_alerts_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export Alert Report", data=csv, 
                           file_name=f"soc_alerts_day{st.session_state.current_day}.csv",
                           mime='text/csv', use_container_width=True)

    if st.button("🗑️ Reset Simulation", use_container_width=True):
        st.session_state.clear()
        st.cache_resource.clear()
        st.rerun()

# --- 6. HEADER & METRICS ---
st.title("AI-Powered SOC Analyst Dashboard")
st.caption("Monitoring LANL Network Authentication Events for Anomalous Sequences")

# Calculate current sim time for filtering ground truth
current_sim_timestamp = ((st.session_state.current_day - 1) * 86400) + (st.session_state.current_hour * 3600)

if not red_ground_truth.empty:
    occurred_red_events = red_ground_truth[red_ground_truth['Timestamp'] < current_sim_timestamp]
    total_occurred = len(occurred_red_events)
else:
    total_occurred = 0

m1, m2, m3, m4 = st.columns(4)
total_anomalies = len(st.session_state.total_alerts_df)
total_hits = sum([h['red_hits'] for h in st.session_state.history]) if st.session_state.history else 0

m1.metric("Timeline", f"Day {st.session_state.current_day} | {st.session_state.current_hour:02d}:00")
m2.metric("Total Anomalies Found", total_anomalies)
m3.metric("Verified Red Team Hits", total_hits)
m4.metric("Events Occurred", total_occurred, help="Red Team events in the data up to this timestamp.")

# --- 7. MAIN PROCESSING BLOCK ---
col_run, col_auto, col_stop = st.columns([2, 1, 1])
with col_run:
    manual_run = st.button(f"🚀 Analyze Hour {st.session_state.current_hour:02d}:00", type="primary", use_container_width=True)
with col_auto:
    if st.button("▶️ Auto-Run", use_container_width=True):
        st.session_state.is_running = True
        st.rerun()
with col_stop:
    if st.button("🛑 Stop", use_container_width=True):
        st.session_state.is_running = False
        st.rerun()

if manual_run or st.session_state.is_running:
    with st.status(f"Scanning Telemetry...", expanded=True) as status:
        raw_df = get_hour_data_fast(st.session_state.current_day, st.session_state.current_hour)

        if not raw_df.empty:
            st.write("🧠 Running Neural Behavioral Analysis...")
            progress_bar = st.progress(0, text="Processing Windows...")
            hour_alerts = sim.process_hour_stream(raw_df, progress_bar=progress_bar)

            # --- Verification & Miss Detection ---
            st.write("🎯 Cross-referencing Red Team Ground Truth...")
            red_hits = sim.verify_red_team(hour_alerts)
            
            # Filter ground truth for THIS specific hour
            start_ts = ((st.session_state.current_day - 1) * 86400) + (st.session_state.current_hour * 3600)
            end_ts = start_ts + 3600
            hour_ground_truth = red_ground_truth[(red_ground_truth['Timestamp'] >= start_ts) & 
                                                 (red_ground_truth['Timestamp'] < end_ts)]
            
            # Find hits we missed in this slice
            missed_events = hour_ground_truth[~hour_ground_truth['Timestamp'].isin(red_hits['Timestamp'])]

            # --- Update History ---
            st.session_state.history.append({
                "Day": st.session_state.current_day, 
                "Hour": st.session_state.current_hour,
                "time_label": f"D{st.session_state.current_day} H{st.session_state.current_hour}",
                "alerts": len(hour_alerts), 
                "red_hits": len(red_hits), 
                "missed": len(missed_events)
            })

            if not hour_alerts.empty:
                st.session_state.total_alerts_df = pd.concat([st.session_state.total_alerts_df, hour_alerts])

            # UI Feedback
            if not red_hits.empty:
                st.error(f"🚨 CRITICAL: {len(red_hits)} Red Team actions confirmed!")
                st.dataframe(red_hits[['Timestamp', 'User', 'Src', 'Dst', 'track']], use_container_width=True)
            
            if not missed_events.empty:
                st.warning(f"⚠️ DETECTION GAP: {len(missed_events)} Red Team events were MISSED in this slice.")
                st.dataframe(missed_events, use_container_width=True)
                
            if red_hits.empty and missed_events.empty:
                st.success(f"Analysis Complete. No threats detected or recorded for this slice.")
        else:
            st.warning(f"No telemetry found for Day {st.session_state.current_day}, Hour {st.session_state.current_hour}.")
        status.update(label=f"SOC Sequence Complete", state="complete", expanded=False)

    # --- Increment Time ---
    if st.session_state.current_hour < 23:
        st.session_state.current_hour += 1
    else:
        st.session_state.current_hour = 0
        st.session_state.current_day += 1
    
    if st.session_state.is_running:
        time.sleep(0.5)
        st.rerun()
    else:
        st.rerun()

# --- 8. VISUALIZATIONS ---
if st.session_state.history:
    st.divider()
    history_df = pd.DataFrame(st.session_state.history)
    tab1, tab2 = st.tabs(["📊 Threat Trends", "🌡️ Anomaly Heatmap"])
    
    with tab1:
        st.subheader("Red Team: Caught vs. Missed")
        chart_data = history_df.melt(id_vars=["time_label"], value_vars=["red_hits", "missed"], 
                                     var_name="Status", value_name="Count")
        st.bar_chart(chart_data, x="time_label", y="Count", color="Status")

    with tab2:
        if not st.session_state.total_alerts_df.empty:
            fig = px.density_heatmap(st.session_state.total_alerts_df, x="Timestamp", y="track", z="lstm_error",
                title="Behavioral Anomaly Intensity", color_continuous_scale="Reds", nbinsx=50)
            st.plotly_chart(fig, use_container_width=True)