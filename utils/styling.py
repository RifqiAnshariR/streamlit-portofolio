import streamlit as st

def load_css():
    with open(r'static\css\style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
