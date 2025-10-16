import streamlit as st
import numpy as np
import pandas as pd
import pickle

st.set_page_config(page_title="Breast Cancer Prediction", layout="centered")

st.title("Breast Cancer Prediction")
st.write("Enter the tumor features to predict whether the tumor is malignant or benign using the trained Random Forest model.")

# Feature names from sklearn.datasets.load_breast_cancer().feature_names
FEATURE_NAMES = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
    'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
    'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

# Load default means from sklearn dataset to prefill inputs
DEFAULTS = {
    'mean radius': 14.127292993630573,
    'mean texture': 19.28964974619291,
    'mean perimeter': 91.9690331838565,
    'mean area': 654.8891040462428,
    'mean smoothness': 0.0963603317320078,
    'mean compactness': 0.10434075886844418,
    'mean concavity': 0.08879931823474873,
    'mean concave points': 0.04891977888171859,
    'mean symmetry': 0.1811621836950141,
    'mean fractal dimension': 0.06279822016308651,
    'radius error': 0.40517296599980863,
    'texture error': 1.2168533582089556,
    'perimeter error': 2.866059477124182,
    'area error': 40.33707923880597,
    'smoothness error': 0.00640541345451486,
    'compactness error': 0.01803263745033113,
    'concavity error': 0.031731199999999998,
    'concave points error': 0.011796095238095238,
    'symmetry error': 0.020542093952830188,
    'fractal dimension error': 0.003795564045267489,
    'worst radius': 16.269190176470587,
    'worst texture': 25.67722352941176,
    'worst perimeter': 107.2612137254902,
    'worst area': 880.5830588235294,
    'worst smoothness': 0.13237176470588237,
    'worst compactness': 0.25426588235294115,
    'worst concavity': 0.2721882352941177,
    'worst concave points': 0.11460647058823529,
    'worst symmetry': 0.2900758823529412,
    'worst fractal dimension': 0.08394678431372549
}

st.sidebar.header("Input features")
user_inputs = {}
for feat in FEATURE_NAMES:
    default_val = DEFAULTS.get(feat, 0.0)
    user_inputs[feat] = st.sidebar.number_input(feat, value=float(default_val))

if st.sidebar.button("Predict"):
    X = pd.DataFrame([user_inputs])[FEATURE_NAMES]

    # Load model
    try:
        with open("breast_cancer_rf_model.pkl", "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        st.error("Model file 'breast_cancer_rf_model.pkl' not found in the app folder.")
        st.stop()

    pred = model.predict(X)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0][1]

    label = "Malignant" if pred == 0 or pred == 1 and pred == 0 else "Benign"
    # Note: sklearn breast cancer target: 0=malignant, 1=benign typically; adjust message
    if pred == 0:
        label = "Malignant"
    else:
        label = "Benign"

    st.subheader("Prediction")
    st.write(f"Predicted label: {label} (model output: {pred})")
    if proba is not None:
        st.write(f"Probability of being malignant: {proba:.3f}")

    # Explain with feature importances if available
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        imp_df = pd.DataFrame({"feature": FEATURE_NAMES, "importance": importances})
        imp_df = imp_df.sort_values("importance", ascending=False).head(10)
        st.subheader("Top feature importances")
        st.table(imp_df.set_index("feature"))

    # Offer to download result
    result = X.copy()
    result["prediction"] = label
    if proba is not None:
        result["probability_malignant"] = proba
    st.download_button("Download result as CSV", result.to_csv(index=False), file_name="prediction.csv")
else:
    st.info("Adjust input features in the sidebar and click Predict.")
