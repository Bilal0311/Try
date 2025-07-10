import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = xgb.XGBClassifier()
model.load_model("models/xgb_model.json")

# App title
st.title("Customer Churn Prediction App")

st.markdown("""
Enter customer details below to predict whether they are likely to churn.
""")

# Inputs
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
total_charges = st.number_input("Total Charges", min_value=0.0, value=1000.0)
senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])

gender = st.selectbox("Gender", ["Male", "Female"])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["No internet service", "No", "Yes"])
online_backup = st.selectbox("Online Backup", ["No internet service", "No", "Yes"])
device_protection = st.selectbox("Device Protection", ["No internet service", "No", "Yes"])
tech_support = st.selectbox("Tech Support", ["No internet service", "No", "Yes"])
streaming_tv = st.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
streaming_movies = st.selectbox("Streaming Movies", ["No internet service", "No", "Yes"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox("Payment Method", [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)"
])

# Encode Senior Citizen
senior_citizen_encoded = 1 if senior_citizen == "Yes" else 0

# Create a raw DataFrame with a single row
raw_input_df = pd.DataFrame({
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges],
    "SeniorCitizen": [senior_citizen_encoded],
    "gender": [gender],
    "Partner": [partner],
    "Dependents": [dependents],
    "PhoneService": [phone_service],
    "MultipleLines": [multiple_lines],
    "InternetService": [internet_service],
    "OnlineSecurity": [online_security],
    "OnlineBackup": [online_backup],
    "DeviceProtection": [device_protection],
    "TechSupport": [tech_support],
    "StreamingTV": [streaming_tv],
    "StreamingMovies": [streaming_movies],
    "Contract": [contract],
    "PaperlessBilling": [paperless_billing],
    "PaymentMethod": [payment_method]
})

# One-hot encode the input
input_encoded = pd.get_dummies(raw_input_df)

# Load your model's expected columns
expected_cols = model.get_booster().feature_names if hasattr(model, "get_booster") else model.feature_names_in_

# Ensure all expected columns exist
for col in expected_cols:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

# Reorder columns to match the model's training data
input_encoded = input_encoded[expected_cols]

if st.button("Predict Churn"):
    prediction = model.predict(input_encoded)
    proba = model.predict_proba(input_encoded)[0][1]
    st.write("Prediction:", "Churn" if prediction[0]==1 else "Not Churn")
    st.write(f"Churn Probability: {proba:.2f}")
