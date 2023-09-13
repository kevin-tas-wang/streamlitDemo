import streamlit as st

st.set_page_config(layout="wide", page_title="Gym Injurey Prediction Testing")

st.write("## Find the right model for Gym Injurey Prediction")
st.write(
    ":dog: Try uploading some test data to explore different models /settings in prediction :grin:"
)
st.sidebar.write("## Upload a sample data  :gear:")

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload a sample file", type=["csv"])

if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error("The uploaded file is too large. Please upload a file smaller than 5MB.")
    else:
        st.write("You have uploaded successfully!")
else:
    st.write("Try uploading something :grin:")
