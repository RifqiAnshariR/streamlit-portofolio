import streamlit as st
from config.config import Config
from utils.styling import load_css


st.set_page_config(
    page_title="Home", 
    page_icon=Config.PAGE_ICON, 
    layout=Config.LAYOUT, 
    menu_items=Config.MENU_ITEMS
)


load_css()


st.title("Streamlit Portofolio")
st.subheader("Welcome...")
st.write("This site showcases projects in machine learning and data science, deployed using Streamlit and FastAPI.")


st.html("<div class='footer'>©2025 Rifqi Anshari Rasyid.</div>")
