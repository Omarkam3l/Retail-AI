"""Home page with overview cards and summary statistics."""
import streamlit as st
from dashboard.api_client import APIClient


def render():
    st.markdown("# 🏠 Dashboard Overview")
    st.markdown("---")

    client = APIClient()
    status = client.system_status()

    if "error" in status:
        st.warning(f"⚠️ API Status: {status.get('error', 'Unknown')}")
        status = {}

    # Summary cards row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🟢 Active Cameras", status.get("active_cameras", 0))
    with col2:
        st.metric("🚨 Total Alerts", status.get("total_alerts", 0))
    with col3:
        st.metric("💻 CPU Usage", f"{status.get('cpu_percent', 0):.1f}%")
    with col4:
        st.metric("🧠 Memory Usage", f"{status.get('memory_percent', 0):.1f}%")

    st.markdown("---")

    # Pipeline metrics
    metrics = client.get_metrics()
    if "error" not in metrics:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📹 Pipeline FPS", f"{metrics.get('pipeline_fps', 0):.1f}")
        with col2:
            st.metric("⏱️ Avg Latency", f"{metrics.get('avg_latency_ms', 0):.1f} ms")
        with col3:
            st.metric("🔍 Total Detections", metrics.get("total_detections", 0))

    # System info
    st.markdown("### System Information")
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Status**: {status.get('status', 'N/A')}")
        st.info(f"**GPU Available**: {'✅ Yes' if status.get('gpu_available') else '❌ No'}")
    with col2:
        uptime = status.get("uptime_seconds", 0)
        hours = int(uptime // 3600)
        minutes = int((uptime % 3600) // 60)
        st.info(f"**Uptime**: {hours}h {minutes}m")
        st.info(f"**GPU Memory**: {status.get('gpu_memory_percent', 0):.1f}%")
