"""Settings and configuration page."""
import streamlit as st
from dashboard.api_client import APIClient


def render():
    st.markdown("# ⚙️ Settings")
    st.markdown("---")

    client = APIClient()

    # Register new camera
    st.markdown("### Register Camera")
    with st.form("register_camera"):
        cam_id = st.text_input("Camera ID", value="cam_01")
        source = st.text_input("Source (file path, RTSP URL, or webcam index)", value="0")
        confidence = st.slider("Detection Confidence", 0.1, 1.0, 0.35)
        submitted = st.form_submit_button("Register Camera")

        if submitted:
            result = client.register_camera(cam_id, source, confidence)
            if "error" in result:
                st.error(f"Failed: {result['error']}")
            else:
                st.success(f"Camera '{cam_id}' registered successfully!")

    st.markdown("---")

    # Detection settings
    st.markdown("### Detection Settings")
    col1, col2 = st.columns(2)
    with col1:
        st.selectbox("Model", ["yolo11s.pt", "yolo11m.pt", "yolo11l.pt"])
        st.selectbox("Device", ["auto", "cpu", "cuda"])
    with col2:
        st.number_input("Max FPS", min_value=1, max_value=120, value=30)
        st.number_input("Batch Size", min_value=1, max_value=32, value=1)

    st.markdown("---")

    # Alert settings
    st.markdown("### Alert Settings")
    st.slider("Risk Threshold (Medium)", 0.0, 1.0, 0.5)
    st.slider("Risk Threshold (High)", 0.0, 1.0, 0.7)
    st.number_input("Alert Cooldown (seconds)", min_value=1, max_value=300, value=60)
