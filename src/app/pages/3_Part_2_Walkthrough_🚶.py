from pathlib import Path
import streamlit as st

st.sidebar.markdown("# Notebook for Programming Project")
st.sidebar.markdown("This is a markdown version of the exploration notebook found on the root directory of this poject.")

exploration_nb = Path("exploration.md").read_text()

st.markdown(exploration_nb, unsafe_allow_html=True)