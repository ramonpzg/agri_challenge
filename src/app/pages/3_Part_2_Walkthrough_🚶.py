from pathlib import Path
import streamlit as st

st.sidebar.markdown("# Notebook with the Project")

exploration_nb = Path("exploration.md").read_text()

st.markdown(exploration_nb, unsafe_allow_html=True)