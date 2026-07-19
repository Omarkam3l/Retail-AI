"""Pipeline metrics visualization."""
import streamlit as st
from dashboard.api_client import APIClient


def render():
    st.markdown("# 📈 Pipeline Metrics")
    st.markdown("---")

    client = APIClient()
    metrics = client.get_metrics()

    if "error" in metrics:
        st.warning(f"⚠️ {metrics['error']}")
        return

    # Summary cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Pipeline FPS", f"{metrics.get('pipeline_fps', 0):.1f}")
    with col2:
        st.metric("Avg Latency", f"{metrics.get('avg_latency_ms', 0):.1f} ms")
    with col3:
        st.metric("Total Frames", metrics.get("total_frames_processed", 0))
    with col4:
        st.metric("Total Detections", metrics.get("total_detections", 0))

    st.markdown("---")

    # Stage latencies
    stage_latencies = metrics.get("stage_latencies", {})
    if stage_latencies:
        st.markdown("### Stage Latencies (ms)")
        import pandas as pd
        df = pd.DataFrame(list(stage_latencies.items()), columns=["Stage", "Latency (ms)"])
        st.bar_chart(df.set_index("Stage"))
    else:
        st.info("No stage latency data available yet.")
