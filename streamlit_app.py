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
senior_citizen = st.radio("Senior Citizen", ["Yes", "No"])
partner = st.radio("Partner", ["Yes", "No"])
dependents = st.radio("Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)


st.divider()
st.header("Phone Service")

phone_service = st.radio("Has Phone Service", ["Yes", "No"])
if phone_service == "Yes":
    multiple_lines = st.radio("Multiple Lines", ["Yes", "No"])
else:
    multiple_lines = "No phone service"


st.divider()
st.header("Internet Service")

internet_service = st.radio("Has Internet Service", ["DSL", "Fiber optic", "No"])

if internet_service == "No":
    online_security = "No internet service"
    online_backup = "No internet service"
    device_protection = "No internet service"
    tech_support = "No internet service"
    streaming_tv = "No internet service"
    streaming_movies = "No internet service"
else:

    st.write("Internet Service Add-ons")
    online_security = "Yes" if st.checkbox("Online Security") else "No"
    online_backup = "Yes" if st.checkbox("Online Backup") else "No"
    device_protection = "Yes" if st.checkbox("Device Protection") else "No"
    tech_support = "Yes" if st.checkbox("Tech Support") else "No"
    streaming_tv = "Yes" if st.checkbox("Streaming TV") else "No"
    streaming_movies = "Yes" if st.checkbox("Streaming Movies") else "No"


st.divider()
st.header("Billing")

paperless_billing = st.radio("Paperless Billing", ["Yes", "No"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
payment_method = st.selectbox("Payment Method", [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)"
])

monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=500.0, value=60.0)
total_charges = st.number_input("Total Charges")

manual_mappings = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
    # "Churn": {"No": 0, "Yes": 1}
}

onehot_apply_on = ["MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod", ]


if st.button("Predict"):
    x_input = pd.Series({
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    })

    for col, mapping in manual_mappings.items():
        x_input[col] = x_input[col].map(mapping)

    x_input = pd.get_dummies(x_input, columns=onehot_apply_on, dtype=int)
    x_input[["tenure"]] = minmax.transform(x_input[["tenure"]])
    x_input[["TotalCharges", "MonthlyCharges"]] = yeo.transform(x_input[["TotalCharges", "MonthlyCharges"]])

    random_forest_prob = random_forest.predict_proba(x_input)
    gradient_boosting_prob = gradient_boosting.predict_proba(x_input)
    svc_prob = svc.predict_proba(x_input)

    st.write(f"Model 1 (Random Forest): {random_forest_prob:.4f}")
    st.write(f"Model 2 (Gradient Boosting): {gradient_boosting_prob:.4f}")
    st.write(f"Model 3 (Logistic Regression): {svc_prob:.4f}")    

    stacked_input = np.array([[random_forest_prob, gradient_boosting_prob, svc_prob]])

    meta_model_pred = meta_model.predict(stacked_input)

    st.write(f"Final verdict: {meta_model_pred}")
