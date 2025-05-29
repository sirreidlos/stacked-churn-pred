import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt

# --- Load pretrained models ---
@st.cache_resource
def load_models():
    return {
        "random_forest": joblib.load('random_forest.pkl'),
        "gradient_boosting": joblib.load('gradient_boosting.pkl'),
        "svc": joblib.load('svc.pkl'),
        "meta_model": joblib.load('meta_model_logreg.pkl'),
    }

@st.cache_data
def load_stacked_features():
    return pd.read_csv("./stacked_features.csv")

@st.cache_resource
def load_scalers():
    scaler = joblib.load("scaler.pkl")
    return scaler["minmax"], scaler["yeo"]

@st.cache_resource
def get_explainer(_model, data_sample):
    return shap.KernelExplainer(_model.predict_proba, shap.kmeans(data_sample, 10))

# Load everything with caching
models = load_models()
random_forest = models["random_forest"]
gradient_boosting = models["gradient_boosting"]
svc = models["svc"]
meta_model = models["meta_model"]

stacked_features = load_stacked_features()
minmax, yeo = load_scalers()

explainer = get_explainer(meta_model, stacked_features)

weights = abs(meta_model.coef_[0])
bias = meta_model.intercept_[0]
feature_labels = [
    "rf_0", "rf_1",
    "gb_0", "gb_1",
    "svc_0", "svc_1"
]

# --- UI ---
st.title("Stacking Classifier Demo")

st.markdown('''
This is an application demonstrating customer churn prediction
of a telecommunications company. It uses stacking ensemble method
with random forest, gradient boosting, and SVM as it's base models,
and logarithmic regression as it's meta model, allowing us direct
access to the meta model's weights too.

            
Members:
1. 2702324745 - Erik Sanjaya
2. 2702242713 - Anthony Gilles Rudolfo
3. 2702338334 - Joceline Araki
''')

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
    container = st.container(border=True)
    container.write("Internet Service Add-ons")
    online_security = "Yes" if container.checkbox("Online Security") else "No"
    online_backup = "Yes" if container.checkbox("Online Backup") else "No"
    device_protection = "Yes" if container.checkbox("Device Protection") else "No"
    tech_support = "Yes" if container.checkbox("Tech Support") else "No"
    streaming_tv = "Yes" if container.checkbox("Streaming TV") else "No"
    streaming_movies = "Yes" if container.checkbox("Streaming Movies") else "No"


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

st.divider()

help_text = '''The threshold slider can be interpreted as how willing
you are as to accept a customer churning, with a lower value meaning 
less willing. This setting is important since finding new customers 
cost more than retaining existing ones, and balancing between the two
is also very important.'''

# st.markdown('''
# The threshold slider can be interpreted as how willing you are as to accept a
# customer churning, with a lower value meaning less willing. This setting
# is important since finding new customers cost more than retaining
# existing ones.
# ''')
threshold = st.slider("Threshold", help=help_text, min_value=0.0, max_value=1.0, value=0.5)

if st.button("Predict"):
    data = {
        "gender": 1 if gender == "Male" else 0,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": 1 if partner == "Yes" else 0,
        "Dependents": 1 if dependents == "Yes" else 0,
        "tenure": float(tenure),
        "PhoneService": 1 if phone_service == "Yes" else 0,
        "PaperlessBilling": 1 if paperless_billing == "Yes" else 0,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    data.update({
        "MultipleLines_No": 1 if multiple_lines == "No" else 0,
        "MultipleLines_No phone service": 1 if multiple_lines == "No phone service" else 0,
        "MultipleLines_Yes": 1 if multiple_lines == "Yes" else 0,
    })

    data.update({
        "InternetService_DSL": 1 if internet_service == "DSL" else 0,
        "InternetService_Fiber optic": 1 if internet_service == "Fiber optic" else 0,
        "InternetService_No": 1 if internet_service == "No" else 0,
    })

    data.update({
        "OnlineSecurity_No": 1 if online_security == "No" else 0,
        "OnlineSecurity_No internet service": 1 if online_security == "No internet service" else 0,
        "OnlineSecurity_Yes": 1 if online_security == "Yes" else 0,
    })

    data.update({
        "OnlineBackup_No": 1 if online_backup == "No" else 0,
        "OnlineBackup_No internet service": 1 if online_backup == "No internet service" else 0,
        "OnlineBackup_Yes": 1 if online_backup == "Yes" else 0,
    })

    data.update({
        "DeviceProtection_No": 1 if device_protection == "No" else 0,
        "DeviceProtection_No internet service": 1 if device_protection == "No internet service" else 0,
        "DeviceProtection_Yes": 1 if device_protection == "Yes" else 0,
    })

    data.update({
        "TechSupport_No": 1 if tech_support == "No" else 0,
        "TechSupport_No internet service": 1 if tech_support == "No internet service" else 0,
        "TechSupport_Yes": 1 if tech_support == "Yes" else 0,
    })

    data.update({
        "StreamingTV_No": 1 if streaming_tv == "No" else 0,
        "StreamingTV_No internet service": 1 if streaming_tv == "No internet service" else 0,
        "StreamingTV_Yes": 1 if streaming_tv == "Yes" else 0,
    })

    data.update({
        "StreamingMovies_No": 1 if streaming_movies == "No" else 0,
        "StreamingMovies_No internet service": 1 if streaming_movies == "No internet service" else 0,
        "StreamingMovies_Yes": 1 if streaming_movies == "Yes" else 0,
    })

    data.update({
        "Contract_Month-to-month": 1 if contract == "Month-to-month" else 0,
        "Contract_One year": 1 if contract == "One year" else 0,
        "Contract_Two year": 1 if contract == "Two year" else 0,
    })

    data.update({
        "PaymentMethod_Bank transfer (automatic)": 1 if payment_method == "Bank transfer (automatic)" else 0,
        "PaymentMethod_Credit card (automatic)": 1 if payment_method == "Credit card (automatic)" else 0,
        "PaymentMethod_Electronic check": 1 if payment_method == "Electronic check" else 0,
        "PaymentMethod_Mailed check": 1 if payment_method == "Mailed check" else 0,
    })

    x_input = pd.DataFrame(data, index=[0])
    x_input[["tenure"]] = minmax.transform(x_input[["tenure"]])
    x_input[["TotalCharges", "MonthlyCharges"]] = yeo.transform(x_input[["TotalCharges", "MonthlyCharges"]])

    def format_prediction(position, name, probs, pred):
        label = "will" if pred[0] == 1 else "will not"
        confidence = probs[0][pred[0]] * 100
        color = "red" if pred[0] == 1 else "green"
        if pred[0] == 0:
            position.metric(name, "Will Not Churn", f"{confidence:.2f}% confidence")
        else:
            position.metric(name, "Will Churn", f"{confidence:.2f}% confidence", delta_color="inverse")
        # return f"<span style='color:{color}'>{name} thinks the customer {label} churn with {confidence:.2f}% confidence.</span>"


    random_forest_prob = random_forest.predict_proba(x_input)
    random_forest_pred = random_forest.predict(x_input)

    gradient_boosting_prob = gradient_boosting.predict_proba(x_input)
    gradient_boosting_pred = gradient_boosting.predict(x_input)

    svc_prob = svc.predict_proba(x_input)
    svc_pred = svc.predict(x_input)

    a, b, c = st.columns(3)


    stacked_input = np.hstack((random_forest_prob, gradient_boosting_prob, svc_prob))

    meta_model_prob = meta_model.predict_proba(stacked_input)
    meta_model_pred = meta_model.predict(stacked_input)
    meta_label = "will" if meta_model_pred[0] == 1 else "will not"
    meta_confidence = meta_model_prob[0][meta_model_pred[0]] * 100
    meta_color = "red" if meta_model_pred[0] == 1 else "green"
    prob_for_churn = meta_model_prob[:, 1][0]
    # st.write(type(prob_for_churn))

    if prob_for_churn <= threshold:
        st.metric("Final Verdict", "Will Not Churn", f"{(1-prob_for_churn) * 100:.2f}% confidence", border=True)
    else:
        st.metric("Final Verdict", "Will Churn", f"{prob_for_churn*100:.2f}% confidence", delta_color="inverse", border=True)
    


    with st.expander("Explanation"):
        st.markdown("""
        ### 🔍 How the Prediction Works

        We use a method called **stacking** to make a more accurate prediction.

        Think of it like asking three different experts (models) for their opinions.  
        Each expert gives a **probability** for whether a customer will churn.

        Then, a **final decision-maker** (called a *meta model*) listens to these experts and makes the final call — but it doesn’t treat all opinions equally. It has learned over time which expert tends to be more reliable and **gives them more weight**.

        ---
        ### 🤖 What Each Expert Said

        Below are the predictions from each model:
        """)

        a, b, c = st.columns(3)
        format_prediction(a, "Gradient Boosting", gradient_boosting_prob, gradient_boosting_pred)
        format_prediction(b, "Random Forest", random_forest_prob, random_forest_pred)
        format_prediction(c, "SVC", svc_prob, svc_pred)

        st.divider()
        st.subheader("📊 How Much Each Expert Influenced the Final Decision")

        selected_features = [
            "rf_1" if random_forest_pred[0] == 1 else "rf_0",
            "gb_1" if gradient_boosting_pred[0] == 1 else "gb_0",
            "svc_1" if svc_pred[0] == 1 else "svc_0"
        ]
        selected_indices = [feature_labels.index(f) for f in selected_features]
        selected_weights = [weights[i] for i in selected_indices]
        selected_labels = [feature_labels[i] for i in selected_indices]

        df = pd.DataFrame({
            "Feature": selected_labels,
            "Weight": selected_weights
        })
        st.bar_chart(df.set_index("Feature"))

        st.divider()
        st.subheader("🧮 Putting It All Together")

        st.markdown("""
        Here's how the final score is calculated:

        1. Take the probability each expert gave.
        2. Multiply it by how much we trust that expert (the weight).
        3. Add everything together with a small bias term.
        4. Pass it through a formula (called a **logistic function**) to get the final probability.

        **Formula (simplified):**  
        `Final Score = (RF prob × RF weight) + (GB prob × GB weight) + (SVC prob × SVC weight) + bias`

        This score is converted into a probability like this:

        """, unsafe_allow_html=True)

        st.latex(r"""
            \text{logit}(x) = w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b
        """)
        st.latex(r"""
            \text{Probability} = \frac{1}{1 + e^{-(\text{logit}(x))}}
        """)

        st.markdown(f"""
        ---
        ### 📈 Final Result

        - **Predicted chance of churn**: **{prob_for_churn * 100:.2f}%**
        - **Churn threshold**: **{threshold * 100:.2f}%**

        If the probability is above the threshold, the customer is predicted to **churn**. Otherwise, they are predicted to **stay**.
        """)