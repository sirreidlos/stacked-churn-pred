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

gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)

PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["No internet service", "No", "Yes"])
OnlineBackup = st.selectbox("Online Backup", ["No internet service", "No", "Yes"])
DeviceProtection = st.selectbox("Device Protection", ["No internet service", "No", "Yes"])
TechSupport = st.selectbox("Tech Support", ["No internet service", "No", "Yes"])
StreamingTV = st.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
StreamingMovies = st.selectbox("Streaming Movies", ["No internet service", "No", "Yes"])

Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox("Payment Method", [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)"
])

MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=500.0, value=70.0)
TotalCharges = st.text_input("Total Charges")  # Kept as text since original column is object type

# # Example input fields for features
# feature1 = st.number_input("Feature 1", value=0.0)
# feature2 = st.number_input("Feature 2", value=0.0)
# feature3 = st.number_input("Feature 3", value=0.0)

# if st.button("Predict"):
#     X_input = np.array([[feature1, feature2, feature3]])

#     # Step 1: Get each model's predict_proba
#     prob1 = model1.predict_proba(X_input)[0][1]
#     prob2 = model2.predict_proba(X_input)[0][1]
#     prob3 = model3.predict_proba(X_input)[0][1]

#     st.subheader("Base Model Probabilities (Positive Class):")
#     st.write(f"Model 1 (Random Forest): {prob1:.4f}")
#     st.write(f"Model 2 (Gradient Boosting): {prob2:.4f}")
#     st.write(f"Model 3 (Logistic Regression): {prob3:.4f}")

#     # Step 2: Stack these probs and predict using meta-model
#     stacked_input = np.array([[prob1, prob2, prob3]])

#     if isinstance(stack_model, StackingClassifier):
#         final_pred = stack_model.predict(X_input)[0]
#         final_proba = stack_model.predict_proba(X_input)[0][1]
#     else:
#         # Custom stacking model using base model outputs as features
#         final_pred = stack_model.predict(stacked_input)[0]
#         final_proba = stack_model.predict_proba(stacked_input)[0][1]

#     st.subheader("Final Stacking Model Output:")
#     st.write(f"Prediction: {'Positive' if final_pred else 'Negative'}")
#     st.write(f"Probability: {final_proba:.4f}")

#     # Step 3: Show stacking weights (if meta-model is linear)
#     if hasattr(stack_model, 'final_estimator_'):
#         meta_model = stack_model.final_estimator_
#     else:
#         meta_model = stack_model

#     if hasattr(meta_model, 'coef_'):
#         st.subheader("Meta-Model Weights for Base Models:")
#         weights = meta_model.coef_[0]
#         weight_df = pd.DataFrame({
#             "Base Model": ["Model 1", "Model 2", "Model 3"],
#             "Weight": weights
#         })
#         st.dataframe(weight_df.style.format({"Weight": "{:.4f}"}))
#     elif hasattr(meta_model, 'feature_importances_'):
#         st.subheader("Meta-Model Feature Importances:")
#         importances = meta_model.feature_importances_
#         imp_df = pd.DataFrame({
#             "Base Model": ["Model 1", "Model 2", "Model 3"],
#             "Importance": importances
#         })
#         st.dataframe(imp_df.style.format({"Importance": "{:.4f}"}))
#     else:
#         st.write("Meta-model does not expose interpretable weights.")
