"""Dark theme CSS and custom styling for the dashboard."""

DARK_THEME_CSS = """
<style>
    /* Main dark theme */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1d23;
        border-right: 1px solid #2d333b;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1d23, #2d333b);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #58a6ff;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Alert badges */
    .alert-critical {
        background-color: #f85149;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
    }
    
    .alert-high {
        background-color: #d29922;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
    }
    
    .alert-medium {
        background-color: #3fb950;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
    }
    
    .alert-low {
        background-color: #388bfd;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
    }
    
    /* Status indicators */
    .status-active {
        color: #3fb950;
        font-weight: 600;
    }
    
    .status-inactive {
        color: #8b949e;
    }
    
    .status-error {
        color: #f85149;
        font-weight: 600;
    }
    
    /* Header styling */
    .dashboard-header {
        background: linear-gradient(90deg, #0d1117 0%, #161b22 100%);
        padding: 20px 30px;
        border-bottom: 2px solid #30363d;
        margin-bottom: 20px;
    }
    
    .dashboard-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #58a6ff;
        margin: 0;
    }
    
    /* Tables */
    .dataframe {
        background-color: #161b22 !important;
        color: #c9d1d9 !important;
    }
    
    /* Gauge container */
    .gauge-container {
        text-align: center;
        padding: 10px;
    }
</style>
"""


def inject_styles():
    """Injects custom CSS into the Streamlit page."""
    import streamlit as st
    st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)
