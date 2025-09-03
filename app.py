import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and scaler
model = joblib.load("breast_cancer_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Breast Cancer Prediction", layout="centered")
st.title("🩺 Breast Cancer Prediction App")
st.write("This app predicts whether a tumor is **Benign** or **Malignant** based on patient features.")

# Example feature inputs (use the same order as your training dataset)
# Adjust these feature names based on what your notebook used (X_train columns)
feature_names = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean"
]

# Sidebar for input mode
mode = st.sidebar.radio("Choose input mode:", ["Single Patient", "Batch Prediction"])

if mode == "Single Patient":
    st.subheader("🔹 Enter Patient Data")

    user_input = []
    for feature in feature_names:
        val = st.number_input(f"Enter {feature}:", min_value=0.0, step=0.1)
        user_input.append(val)

    if st.button("Predict"):
        # Scale and predict
        input_scaled = scaler.transform([user_input])
        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]

        if prediction == 1:
            st.error(f"⚠️ Prediction: **Malignant (Cancer Detected)** — Probability {proba[1]*100:.2f}%")
        else:
            st.success(f"✅ Prediction: **Benign (No Cancer)** — Probability {proba[0]*100:.2f}%")

elif mode == "Batch Prediction":
    st.subheader("📂 Upload CSV File")
    uploaded_file = st.file_uploader("Upload a CSV with patient data", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Scale and predict
        data_scaled = scaler.transform(data)
        preds = model.predict(data_scaled)
        probas = model.predict_proba(data_scaled)

        data["Prediction"] = ["Malignant" if p == 1 else "Benign" for p in preds]
        data["Malignant_Probability"] = probas[:, 1]
        data["Benign_Probability"] = probas[:, 0]

        st.write("### 🔎 Results")
        st.dataframe(data)

        # Show summary chart
        st.bar_chart(data["Prediction"].value_counts())
