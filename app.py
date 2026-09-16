# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow import keras

# ============================================================
# LOAD SAVED ARTIFACTS
# ============================================================
@st.cache_resource
def load_artifacts():
    model = keras.models.load_model('diabetes_mlp.keras')
    scaler = joblib.load('diabetes_scaler.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
    return model, scaler, feature_columns

model, scaler, feature_columns = load_artifacts()

st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="centered")
st.title("🩺 Diabetes Risk Prediction")
st.caption(
    "This tool is a screening aid based on an MLP trained on the Pima Indians Diabetes dataset. "
    "It is **not** a medical diagnosis. Consult a healthcare professional for clinical decisions."
)

# ============================================================
# USER INPUT
# ============================================================
st.subheader("Patient Information")

col1, col2 = st.columns(2)
with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
    glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
with col2:
    insulin = st.number_input("Insulin (mu U/ml)", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
    age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)

# ============================================================
# INPUT VALIDATION
# ============================================================
def validate_inputs(glucose, blood_pressure, bmi):
    warnings = []
    if glucose == 0:
        warnings.append("Glucose of 0 is not physiologically possible — please enter a valid value.")
    if blood_pressure == 0:
        warnings.append("Blood Pressure of 0 is not physiologically possible — please enter a valid value.")
    if bmi == 0:
        warnings.append("BMI of 0 is not physiologically possible — please enter a valid value.")
    return warnings

# ============================================================
# FEATURE ENGINEERING (must mirror the training pipeline exactly)
# ============================================================
def engineer_features(raw: dict) -> pd.DataFrame:
    row = pd.DataFrame([raw])

    row['AgeGroup'] = pd.cut(row['Age'], bins=[20, 30, 40, 50, 100],
                              labels=['21-30', '31-40', '41-50', '51+'])
    row['BMICategory'] = pd.cut(row['BMI'], bins=[0, 18.5, 25, 30, 100],
                                 labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    row['GlucoseCategory'] = pd.cut(row['Glucose'], bins=[0, 99, 125, 300],
                                     labels=['Normal', 'Prediabetic', 'Diabetic_range'])
    row['Glucose_BMI_interaction'] = row['Glucose'] * row['BMI']
    row['Insulin_log'] = np.log1p(row['Insulin'])

    row = pd.get_dummies(row, columns=['AgeGroup', 'BMICategory', 'GlucoseCategory'], drop_first=True)

    # Align to the exact training-time column set/order — fills any missing dummy columns with 0
    row = row.reindex(columns=feature_columns, fill_value=0)
    return row

# ============================================================
# PREDICTION
# ============================================================
st.divider()

if st.button("Predict", type="primary", use_container_width=True):
    warnings = validate_inputs(glucose, blood_pressure, bmi)

    if warnings:
        for w in warnings:
            st.error(w)
    else:
        raw_input = {
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': blood_pressure,
            'SkinThickness': skin_thickness,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': dpf,
            'Age': age,
        }

        try:
            X_input = engineer_features(raw_input)
            X_scaled = scaler.transform(X_input)

            prob = float(model.predict(X_scaled, verbose=0).ravel()[0])
            prediction = "Diabetic" if prob >= 0.5 else "Non-Diabetic"

            st.subheader("Result")
            if prediction == "Diabetic":
                st.error(f"**Prediction: {prediction}**")
            else:
                st.success(f"**Prediction: {prediction}**")

            st.metric("Probability of Diabetes", f"{prob * 100:.2f}%")
            st.progress(min(max(prob, 0.0), 1.0))

            with st.expander("Show input used for prediction"):
                st.dataframe(X_input)

        except Exception as e:
            st.error(f"Something went wrong while generating a prediction: {e}")
            st.info("Please check that all inputs are valid and try again.")

st.divider()
st.caption(
    "⚠️ This application is for educational purposes only and should not be used as a substitute "
    "for professional medical advice, diagnosis, or treatment."
)