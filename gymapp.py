import streamlit as st

st.set_page_config(layout="wide", page_title="Gym Injurey Prediction Testing")

st.write("## Find the right model for Gym Injurey Prediction")
st.write(
    ":dog: Try uploading some test data to explore different models /settings in prediction :grin:"
)
st.sidebar.write("## Upload a sample data  :gear:")

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
