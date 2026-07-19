"""Main Streamlit dashboard entry point."""
import streamlit as st

# Page config must be first Streamlit command
st.set_page_config(
    page_title="Retail AI Surveillance",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

from dashboard.styles import inject_styles

inject_styles()

# Sidebar navigation
st.sidebar.markdown("## 🔍 Retail AI Surveillance")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📷 Live Cameras", "🚨 Alerts", "📊 Event Timeline",
     "📈 Metrics", "💚 System Health", "🏆 Benchmarks", "⚙️ Settings"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("*v1.0.0 — Production*")

# Route to pages
if page == "🏠 Home":
    from dashboard.pages.home import render
    render()
elif page == "📷 Live Cameras":
    from dashboard.pages.live_cameras import render
    render()
elif page == "🚨 Alerts":
    from dashboard.pages.alerts_page import render
    render()
elif page == "📊 Event Timeline":
    from dashboard.pages.timeline import render
    render()
elif page == "📈 Metrics":
    from dashboard.pages.metrics_page import render
    render()
elif page == "💚 System Health":
    from dashboard.pages.system_health import render
    render()
elif page == "🏆 Benchmarks":
    from dashboard.pages.benchmarks import render
    render()
elif page == "⚙️ Settings":
    from dashboard.pages.settings import render
    render()
