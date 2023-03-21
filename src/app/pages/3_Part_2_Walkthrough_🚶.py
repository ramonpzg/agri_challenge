from pathlib import Path
import streamlit as st

st.sidebar.markdown("# Notebook for Programming Project")

exploration_nb = Path("exploration.md").read_text()

st.markdown(exploration_nb, unsafe_allow_html=True)