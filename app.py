import streamlit as st
import pandas as pd
import joblib
import os

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Heart Disease Prediction System")
st.write("Machine Learning Application for Predicting Heart Disease Risk")

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "heart_disease_model.pkl")

try:
    model = joblib.load(model_path)
except:
    st.error("Model file not found. Place heart_disease_model.pkl in the app folder.")
    st.stop()

# ---------------------------------------------------
# SIDEBAR INPUT FORM
# ---------------------------------------------------
st.sidebar.header("Patient Medical Information")

age = st.sidebar.slider("Age", 20, 100, 45)

sex = st.sidebar.selectbox(
    "Sex",
    ["M", "F"]
)

chest_pain = st.sidebar.selectbox(
    "Chest Pain Type",
    ["ATA", "NAP", "ASY", "TA"]
)

resting_bp = st.sidebar.slider(
    "Resting Blood Pressure",
    80, 200, 120
)

cholesterol = st.sidebar.slider(
    "Cholesterol Level",
    0, 600, 200
)

fasting_bs = st.sidebar.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    [0, 1]
)

resting_ecg = st.sidebar.selectbox(
    "Resting ECG",
    ["Normal", "ST", "LVH"]
)

max_hr = st.sidebar.slider(
    "Maximum Heart Rate",
    60, 220, 150
)

exercise_angina = st.sidebar.selectbox(
    "Exercise Induced Angina",
    ["Y", "N"]
)

oldpeak = st.sidebar.slider(
    "Oldpeak (ST depression)",
    0.0, 6.0, 1.0
)

st_slope = st.sidebar.selectbox(
    "ST Segment Slope",
    ["Up", "Flat", "Down"]
)

# ---------------------------------------------------
# CREATE DATAFRAME
# ---------------------------------------------------
input_data = pd.DataFrame({
    "Age":[age],
    "Sex":[sex],
    "ChestPainType":[chest_pain],
    "RestingBP":[resting_bp],
    "Cholesterol":[cholesterol],
    "FastingBS":[fasting_bs],
    "RestingECG":[resting_ecg],
    "MaxHR":[max_hr],
    "ExerciseAngina":[exercise_angina],
    "Oldpeak":[oldpeak],
    "ST_Slope":[st_slope]
})

st.subheader("Patient Data")
st.write(input_data)

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------
if st.button("Predict Heart Disease Risk"):

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

# ---------------------------------------------------
# INFO SECTION
# ---------------------------------------------------
st.markdown("---")

st.header("About This Model")

st.write(
"""
This machine learning system predicts the likelihood of heart disease
based on clinical features such as:

• Age  
• Chest pain type  
• Blood pressure  
• Cholesterol  
• ECG results  
• Exercise induced angina  

The model analyzes these indicators and provides a risk prediction.

This system is intended for **educational and research purposes only**.
"""
)