import streamlit as st
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load("breast_cancer_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Breast Cancer Prediction", layout="centered")
st.title("🩺 Breast Cancer Prediction App")
st.write("Upload a CSV file with patient data to predict whether tumors are **Benign** or **Malignant**.")

# Expected features (30 columns from dataset)
feature_names = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
]

uploaded_file = st.file_uploader("📂 Upload CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Check if CSV has correct columns
    if list(data.columns) != feature_names:
        st.error("❌ CSV file must have exactly 30 feature columns in the correct order.")
        st.write("Expected columns:", feature_names)
    else:
        # Scale and predict
        data_scaled = scaler.transform(data)
        preds = model.predict(data_scaled)
        probas = model.predict_proba(data_scaled)

        # Add predictions to dataframe
        data["Prediction"] = ["Malignant" if p == 1 else "Benign" for p in preds]
        data["Malignant_Probability"] = probas[:, 1]
        data["Benign_Probability"] = probas[:, 0]

        st.success("✅ Predictions complete!")
        st.dataframe(data)

        # Show summary
        st.subheader("📊 Prediction Summary")
        st.bar_chart(data["Prediction"].value_counts())
