import streamlit as st
import pandas as pd 
from pandas import Interval
from pandas.plotting import scatter_matrix

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import joblib

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

st.write("### Gym Injurey Prediction")
st.write(
    ":dog: This demo uses built-in testing data; in future it will allower users to upload some test data to explore different models/settings in prediction"
)


MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("## Upload a sample data", type=["csv"], disabled=True)

# if my_upload is not None:
#     if my_upload.size > MAX_FILE_SIZE:
#         st.error("The uploaded file is too large. Please upload a file smaller than 5MB.")
#     else:
#         st.write("You have uploaded successfully!")
# else:
#     # st.write("Try uploading something :grin:")
#     st.write("No file uploaded.")

# sidebar
resamplingMethod = st.sidebar.radio(
    "**Under/Over Sampling Method:**",
    ["None", "Random UnderSampling", "Randown Oversampling", "ADASYN Oversampling", "SMOTE Tomek", "SMOTE Enn"],
    index=0)


dimentionMethod = st.sidebar.radio(
    "**Dimensionality reduction Method:**",
    ["None", "PCA", "ICA"],  
    index=0)

mlMethod = st.sidebar.radio(
    "**ML method:**",
    ["Logistics Regression","Random Forest", "Decision Tree", "XGBoost", "Stacking"],  
    index=0)

# get artifact names
tmpModelFileName = f'{resamplingMethod}-{dimentionMethod}-{mlMethod}-ml.joblib'
fittedMethodFileName=f'{resamplingMethod}-{dimentionMethod}-dm.joblib'

# st.write(f'Under/OverSampling method: {resamplingMethod}; Dimensionality reduction method: {dimentionMethod}; ML method:{mlMethod}')
# st.write(tmpModelFileName)
# st.write(fittedMethodFileName )

# load default testing data
rawX_test = pd.read_csv('./testingdata/xtest.csv')
y_testDf = pd.read_csv('./testingdata/ytest.csv')
y_test =y_testDf.after_claim

st.sidebar.write(f'##### using test sample size: {rawX_test.shape}')

# load Dimensionality reduction
if dimentionMethod!="None":
    st.write(fittedMethodFileName)
    filePath= f'./artifacts/{fittedMethodFileName}'
    fittedMethod = joblib.load(filePath)
    X_test=fittedMethod.transform(rawX_test.copy())
else:
     X_test=rawX_test.copy()

# st.sidebar.write(f'##### using test sample size: {X_test.shape}')


# load ml models
# st.write(f'loading {tmpModelFileName} ')
filePath= f'./artifacts/{tmpModelFileName}'
tmpModel = joblib.load(filePath)
st.write(tmpModel)

tmpPredicts = tmpModel.predict(X_test)
tmpCmresults = confusion_matrix(y_test, tmpPredicts)
tmpreport = classification_report(y_test, tmpPredicts)

# st.write(f'Confusion Matrix:\n{tmpCmresults}\n')
st.write(f'Classification Report: \n{tmpreport}\n')

fig = plt.figure(figsize=(3, 1))
sns.heatmap(tmpCmresults, annot=True, fmt='d', linecolor='black', cmap="Greens")
st.pyplot(fig)