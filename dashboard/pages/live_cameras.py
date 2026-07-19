"""Live cameras page with video feeds."""
import streamlit as st
from dashboard.api_client import APIClient


def render():
    st.markdown("# 📷 Live Cameras")
    st.markdown("---")

    client = APIClient()
    cameras = client.list_cameras()

    if not cameras:
        st.info("No cameras registered. Go to Settings to add a camera.")
        return

    for cam in cameras:
        if isinstance(cam, dict):
            with st.expander(f"📷 {cam.get('camera_id', 'Unknown')} — {cam.get('status', 'N/A').upper()}", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Status", cam.get("status", "N/A").upper())
                with col2:
                    st.metric("FPS", f"{cam.get('fps', 0):.1f}")
                with col3:
                    st.metric("Frames", cam.get("frame_count", 0))
                with col4:
                    st.metric("Source", cam.get("source", "N/A")[:30])

                bcol1, bcol2 = st.columns(2)
                with bcol1:
                    if st.button(f"▶️ Start", key=f"start_{cam['camera_id']}"):
                        result = client.start_camera(cam["camera_id"])
                        st.success(result.get("message", "Started"))
                with bcol2:
                    if st.button(f"⏹️ Stop", key=f"stop_{cam['camera_id']}"):
                        result = client.stop_camera(cam["camera_id"])
                        st.info(result.get("message", "Stopped"))
