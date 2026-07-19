"""Event timeline visualization."""
import streamlit as st
from dashboard.api_client import APIClient


def render():
    st.markdown("# 📊 Event Timeline")
    st.markdown("---")

    client = APIClient()
    alerts_data = client.list_alerts(page_size=100)

    if "error" in alerts_data:
        st.warning(f"⚠️ {alerts_data['error']}")
        return

    alerts = alerts_data.get("alerts", [])

    if not alerts:
        st.info("No events to display.")
        return

    st.markdown("### Recent Events")
    for alert in alerts:
        level = alert.get("level", "LOW")
        icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🔵"}.get(level, "⚪")
        timestamp = alert.get("timestamp_ms", 0)
        minutes = int(timestamp / 60000)
        seconds = int((timestamp % 60000) / 1000)

        st.markdown(
            f"{icon} **{level}** — Track {alert.get('track_id', '?')} | "
            f"Camera: {alert.get('camera_id', '?')} | "
            f"Type: {alert.get('event_type', '?')} | "
            f"Time: {minutes}:{seconds:02d}"
        )
