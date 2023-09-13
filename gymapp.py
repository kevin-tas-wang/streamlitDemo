import streamlit as st
import pandas as pd 
from pandas import Interval
from pandas.plotting import scatter_matrix

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder, StandardScaler,RobustScaler
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC 
from xgboost import XGBClassifier

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
    # st.write("Try uploading something :grin:")
    st.write("No file uploaded.")
    
