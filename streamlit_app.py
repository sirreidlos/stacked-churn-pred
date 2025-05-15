import streamlit as st
import numpy as np
import pandas as pd
import joblib


# --- Load pretrained models ---
random_forest = joblib.load('random_forest.pkl')  # Random Forest
gradient_boosting = joblib.load('gradient_boosting.pkl')  # Gradient Boosting
svc = joblib.load('svc.pkl')  # Logistic Regression
meta_model = joblib.load('meta_model_logreg.pkl')  # StackingClassifier or custom meta-model

scaler = joblib.load("scaler.pkl")
minmax = scaler["minmax"]
yeo = scaler["yeo"]

# --- UI ---
st.title("Stacking Classifier Demo")

st.divider()
st.header("Personal Information")

gender = st.radio("Gender", ["Male", "Female"])
SeniorCitizen = st.radio("Senior Citizen", ["Yes", "No"])
Partner = st.radio("Partner", ["Yes", "No"])
Dependents = st.radio("Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)


st.divider()
st.header("Phone Service")

PhoneService = st.radio("Has Phone Service", ["Yes", "No"])
if PhoneService == "Yes":
    MultipleLines = st.radio("Multiple Lines", ["Yes", "No"])
else:
    MultipleLines = "No phone service"


st.divider()
st.header("Internet Service")

InternetService = st.radio("Has Internet Service", ["DSL", "Fiber optic", "No"])

if InternetService == "No":
    OnlineSecurity = "No internet service"
    OnlineBackup = "No internet service"
    DeviceProtection = "No internet service"
    TechSupport = "No internet service"
    StreamingTV = "No internet service"
    StreamingMovies = "No internet service"
else:

    st.write("Internet Service Add-ons")
    OnlineSecurity = "Yes" if st.checkbox("Online Security") else "No"
    OnlineBackup = "Yes" if st.checkbox("Online Backup") else "No"
    DeviceProtection = "Yes" if st.checkbox("Device Protection") else "No"
    TechSupport = "Yes" if st.checkbox("Tech Support") else "No"
    StreamingTV = "Yes" if st.checkbox("Streaming TV") else "No"
    StreamingMovies = "Yes" if st.checkbox("Streaming Movies") else "No"


st.divider()
st.header("Billing")

PaperlessBilling = st.radio("Paperless Billing", ["Yes", "No"])
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaymentMethod = st.selectbox("Payment Method", [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)"
])

MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=500.0, value=60.0)
TotalCharges = st.number_input("Total Charges")

