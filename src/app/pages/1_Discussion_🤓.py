from pathlib import Path
import streamlit as st

st.sidebar.markdown("# Discussion")

discussion = Path("discussion.md").read_text()

st.markdown(discussion, unsafe_allow_html=True)