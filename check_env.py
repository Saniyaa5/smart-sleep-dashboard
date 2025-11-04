import sys
import streamlit as st

st.write("### Python Executable Path")
st.code(sys.executable)

st.write("### Site-Packages Paths")
for p in sys.path:
    st.text(p)
