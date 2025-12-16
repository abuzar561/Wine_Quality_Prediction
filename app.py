import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model and scaler
model = joblib.load('wine_quality_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🍷 Wine Quality Prediction System")
st.write("Enter the chemical values of the wine to predict its quality (1-10 scale).")

# Create input fields for the user
col1, col2, col3 = st.columns(3)

with col1:
    fixed_acidity = st.number_input("Fixed Acidity", max_value=11.5, value=7.90,)
    volatile_acidity = st.number_input("Volatile Acidity", max_value=0.800, value=0.30)
    citric_acid = st.number_input("Citric Acid", max_value=0.65, value=0.40)
    residual_sugar = st.number_input("Residual Sugar", max_value=6.0, value=2.0)

with col2:
    chlorides = st.number_input("Chlorides", max_value=0.415, value=0.060)
    free_sulfur = st.number_input("Free Sulfur Dioxide", max_value=50.0, value=15.0)
    total_sulfur = st.number_input("Total Sulfur Dioxide", max_value=150.0, value=40.0)
    density = st.number_input("Density", max_value=1.000, value=.9960, format="%.4f")

with col3:
    pH = st.number_input("pH", max_value=3.9, value=3.30)
    sulphates = st.number_input("Sulphates", max_value=1.98, value=0.85)
    alcohol = st.number_input("Alcohol", max_value=14.0, value=12.50)

# Create a button to predict
if st.button("Predict Quality"):
    # 1. Prepare the data
    input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, 
                            chlorides, free_sulfur, total_sulfur, density, pH, sulphates, alcohol]])
    
    # 2. Scale the data (using the same scaler as training)
    input_scaled = scaler.transform(input_data)
    
    # 3. Predict
    prediction = model.predict(input_scaled)
    
    # 4. Show Result
    st.success(f"Predicted Wine Quality: {prediction[0]}")
    
    if prediction[0] >= 7:
        st.balloons()
        st.write("🌟 This is a High Quality Wine!")
    elif prediction[0] <= 4:
        st.error("⚠️ This is a Poor Quality Wine.")
    else:
        st.info("😐 This is an Average Quality Wine.")