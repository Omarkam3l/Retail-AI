"""Benchmark results display page."""
import streamlit as st


def render():
    st.markdown("# 🏆 Benchmark Results")
    st.markdown("---")

    st.info("Benchmark results from the evaluation framework will appear here after running experiments.")

    st.markdown("### How to Run Benchmarks")
    st.code("""
# Run from project root:
python -m src.evaluation.benchmark_runner
    """)

    st.markdown("### Expected Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Detection**")
        st.markdown("- mAP@50")
        st.markdown("- mAP@50-95")
        st.markdown("- Precision / Recall")
    with col2:
        st.markdown("**Tracking**")
        st.markdown("- MOTA")
        st.markdown("- MOTP")
        st.markdown("- IDF1")
    with col3:
        st.markdown("**Behavior**")
        st.markdown("- Per-rule F1")
        st.markdown("- Risk accuracy")
        st.markdown("- Alert latency")
