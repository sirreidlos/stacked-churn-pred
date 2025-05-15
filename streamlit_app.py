import streamlit as st
import numpy as np
import pandas as pd
import joblib


# --- Load pretrained models ---
random_forest = joblib.load('random_forest.pkl')  # Random Forest
gradient_boosting = joblib.load('gradient_boosting.pkl')  # Gradient Boosting
svc = joblib.load('svc.pkl')  # Logistic Regression
meta_model = joblib.load('meta_model_logreg.pkl')  # StackingClassifier or custom meta-model

# --- UI ---
st.title("Stacking Classifier Demo")

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
