"""Heart Disease Prediction Web Application.

This Streamlit app predicts heart disease risk using a machine learning model
trained on the UCI Heart Disease dataset.
"""
import os
import pickle
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


@st.cache_resource
def load_model() -> Any:
    """Load the trained model, create it if it doesn't exist.
    
    Returns:
        Trained LogisticRegression model for heart disease prediction.
    """
    model_path = "heart_disease_model.pkl"
    
    if not os.path.exists(model_path):
        st.info("🔄 Setting up the model for first use...")
        
        try:
            # Load the dataset
            df = pd.read_csv("heart-disease.csv")
            
            # Prepare features and target
            X = df.drop(labels="target", axis=1)
            y = df.target.to_numpy()
            
            # Train the model with best parameters
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(C=0.23357214690901212, solver="liblinear")
            model.fit(X, y)
            
            # Save the model
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            
            st.success("✅ Model setup complete!")
            return model
            
        except Exception as e:
            st.error(f"❌ Error setting up model: {str(e)}")
            st.info("Please ensure the heart-disease.csv file exists in the same directory.")
            st.stop()
    
    # Load existing model
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error(f"❌ Model file not found at {model_path}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()


def validate_input(age: int, trestbps: int, chol: int, thalach: int, oldpeak: float) -> bool:
    """Validate user input values are within reasonable medical ranges.
    
    Args:
        age: Patient age in years
        trestbps: Resting blood pressure in mm Hg
        chol: Serum cholesterol in mg/dl
        thalach: Maximum heart rate achieved
        oldpeak: ST depression induced by exercise
        
    Returns:
        True if all inputs are valid, False otherwise
    """
    if not (18 <= age <= 100):
        st.warning("Age should be between 18 and 100 years")
        return False
    if not (80 <= trestbps <= 200):
        st.warning("Resting blood pressure should be between 80 and 200 mm Hg")
        return False
    if not (100 <= chol <= 600):
        st.warning("Cholesterol should be between 100 and 600 mg/dl")
        return False
    if not (70 <= thalach <= 210):
        st.warning("Maximum heart rate should be between 70 and 210 bpm")
        return False
    if not (0.0 <= oldpeak <= 6.0):
        st.warning("ST depression should be between 0.0 and 6.0")
        return False
    return True

# Load the model
model = load_model()

st.title("❤️ Heart Disease Predictor")
st.write("###  Based on Machine Learning Project - Heart Disease Classification")
st.write("This app uses machine learning to predict heart disease risk based on medical data. Enter the patient information below:")

st.sidebar.markdown("### About This App")
st.sidebar.write("Built with Logistic Regression achieving 85% accuracy on the famous heart disease dataset.")

# Input fields with better labels
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (years)", 18, 100, 50)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male", index=1)
    cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], 
                      format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"][x])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], 
                       format_func=lambda x: "No" if x == 0 else "Yes")
    restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2], 
                           format_func=lambda x: ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"][x])

with col2:
    thalach = st.number_input("Max Heart Rate Achieved", 70, 210, 150)
    exang = st.selectbox("Exercise-Induced Angina", options=[0, 1], 
                         format_func=lambda x: "No" if x == 0 else "Yes")
    oldpeak = st.number_input("ST Depression Induced by Exercise", 0.0, 6.0, 1.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[0, 1, 2], 
                         format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
    ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3], 
                        format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect", "Not Described"][x])

# Prepare input
input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                        thalach, exang, oldpeak, slope, ca, thal]])

# Prediction
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.button("🔍 Predict Heart Disease Risk", use_container_width=True):
        # Validate inputs
        if not validate_input(age, trestbps, chol, thalach, oldpeak):
            st.stop()
            
        try:
            prediction = model.predict(input_data)
            probability = model.predict_proba(input_data)[0]
            
            if prediction[0] == 1:
                st.error("🚨 **High Risk of Heart Disease**")
                st.write(f"**Probability: {probability[1]:.1%}**")
                st.write("⚠️ Please consult with a healthcare professional for proper medical evaluation.")
            else:
                st.success("✅ **Low Risk of Heart Disease**")
                st.write(f"**Probability: {probability[0]:.1%}**")
                st.write("💚 This indicates lower risk, but regular health checkups are still recommended.")
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

st.markdown("---")
st.markdown("### ⚕️ Medical Disclaimer")
st.caption("The model was trained on data made using 303 patients, so it is not a reliable diagnostic tool. It is for educational purposes only and should not replace professional medical advice. Always consult with qualified healthcare providers for medical decisions.")