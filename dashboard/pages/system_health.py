"""System health monitoring page."""
import streamlit as st
from dashboard.api_client import APIClient


def render():
    st.markdown("# 💚 System Health")
    st.markdown("---")

    client = APIClient()
    status = client.system_status()

    if "error" in status:
        st.warning(f"⚠️ {status['error']}")
        return

    # CPU & Memory
    col1, col2, col3 = st.columns(3)
    with col1:
        cpu = status.get("cpu_percent", 0)
        st.metric("CPU Usage", f"{cpu:.1f}%")
        st.progress(min(cpu / 100, 1.0))
    with col2:
        mem = status.get("memory_percent", 0)
        st.metric("RAM Usage", f"{mem:.1f}%")
        st.progress(min(mem / 100, 1.0))
    with col3:
        gpu_mem = status.get("gpu_memory_percent", 0)
        st.metric("GPU Memory", f"{gpu_mem:.1f}%")
        st.progress(min(gpu_mem / 100, 1.0))

    st.markdown("---")

    # GPU Info
    st.markdown("### GPU Status")
    if status.get("gpu_available"):
        st.success("✅ GPU Available")
        st.info(f"VRAM Usage: {gpu_mem:.1f}%")
    else:
        st.warning("❌ No GPU detected — running on CPU")

    # System Status
    st.markdown("### Overall Status")
    overall = status.get("status", "unknown")
    if overall == "operational":
        st.success("🟢 All systems operational")
    else:
        st.error(f"🔴 System status: {overall}")

    # Auto refresh
    if st.button("🔄 Refresh"):
        st.rerun()
