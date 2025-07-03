import streamlit as st
import numpy as np
import joblib
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Load saved model and scaler
model = joblib.load('svm_heart_model.pkl')  # or xgb_heart_model.pkl
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("💓 Heart Disease Prediction App")

st.markdown("Enter patient details to predict heart disease risk.")

# Input fields
age = st.number_input("Age", 20, 100, step=1)
sex = st.selectbox("Sex (1 = male, 0 = female)", [1, 0])
cp = st.selectbox("Chest Pain Type (0 = typical angina, 1 = atypical, 2 = non-anginal, 3 = asymptomatic)", [0, 1, 2, 3])
trestbps = st.slider("Resting Blood Pressure", 90, 200, step=1)
chol = st.slider("Cholesterol (mg/dL)", 100, 600, step=1)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True; 0 = False)", [1, 0])
restecg = st.selectbox("Resting ECG (0 = normal, 1 = ST-T wave abnormality, 2 = LVH)", [0, 1, 2])
thalach = st.slider("Max Heart Rate Achieved", 70, 210, step=1)
exang = st.selectbox("Exercise Induced Angina (1 = Yes; 0 = No)", [1, 0])
oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 6.0, step=0.1)
slope = st.selectbox("Slope of ST Segment (0 = upsloping, 1 = flat, 2 = downsloping)", [0, 1, 2])
ca = st.selectbox("No. of Major Vessels (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)", [1, 2, 3])

# Collect input
input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                        thalach, exang, oldpeak, slope, ca, thal]])

# Scale input
input_data_scaled = scaler.transform(input_data)

# Predict probability
proba = model.predict_proba(input_data_scaled)[0][1]

# Apply threshold (tweak this if needed)
threshold = 0.4
prediction = 1 if proba >= threshold else 0

# Display results
st.write(f"🔍 **Predicted Probability of Heart Disease:** `{proba:.2f}`")

if prediction == 1:
    st.error("⚠️ **Heart Disease Detected**. Please consult a medical professional.")
else:
    st.success("✅ **No Heart Disease Detected.** Stay healthy!")

