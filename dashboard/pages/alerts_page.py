"""Alerts page with filtering and details."""
import streamlit as st
import pandas as pd
from dashboard.api_client import APIClient


def render():
    st.markdown("# 🚨 Alerts")
    st.markdown("---")

    client = APIClient()

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        level_filter = st.selectbox("Severity", ["All", "LOW", "MEDIUM", "HIGH", "CRITICAL"])
    with col2:
        page = st.number_input("Page", min_value=1, value=1)
    with col3:
        page_size = st.selectbox("Per page", [25, 50, 100])

    level = level_filter if level_filter != "All" else None
    result = client.list_alerts(page=page, page_size=page_size, level=level)

    if "error" in result:
        st.warning(f"⚠️ {result['error']}")
        return

    alerts = result.get("alerts", [])
    total = result.get("total", 0)
    st.caption(f"Showing {len(alerts)} of {total} alerts")

    if not alerts:
        st.info("No alerts found.")
        return

    df = pd.DataFrame(alerts)
    st.dataframe(df, use_container_width=True, hide_index=True)
