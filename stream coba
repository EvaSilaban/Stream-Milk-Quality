import pickle
import numpy as np
import streamlit as st
import os

# Debugging path file
st.write("Current Directory:", os.getcwd())
st.write("Files in Directory:", os.listdir())

# Check if scikit-learn is installed
try:
    import sklearn
except ModuleNotFoundError:
    st.error("Module 'scikit-learn' not found. Please install it using 'pip install scikit-learn'.")

# Load model and scaler
milkquality = None
scaler = None
try:
    if 'milkquality_model.pkl' in os.listdir():
        with open('milkquality_model.pkl', 'rb') as f:
            milkquality = pickle.load(f)
    else:
        st.error("File 'milkquality_model.pkl' not found.")

    if 'Scaler.pkl' in os.listdir():
        with open('Scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    else:
        st.error("File 'Scaler.pkl' not found.")
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")

# Web app title
st.title("Milk Quality Prediction")

# Input fields for user data
col1, col2 = st.columns(2)
with col1:
    pH = st.text_input("pH")
with col2:
    Temperatur = st.text_input("Temperature")
with col1:
    Rasa = st.text_input("Taste (0=bad, 1=good)")
with col2:
    Bau = st.text_input("Odor (0=bad, 1=good)")
with col1:
    Lemak = st.text_input("Fat (0=bad, 1=good)")
with col2:
    Kekeruhan = st.text_input("Turbidity (0=bad, 1=good)")
with col1:
    Warna = st.text_input("Color")

# Prediction code
if st.button("Predict Milk Quality NOW"):
    try:
        # Ensure all fields are filled
        if all([pH, Temperatur, Rasa, Bau, Lemak, Kekeruhan, Warna]):
            # Convert inputs to float
            pH = float(pH)
            Temperatur = float(Temperatur)
            Rasa = float(Rasa)
            Bau = float(Bau)
            Lemak = float(Lemak)
            Kekeruhan = float(Kekeruhan)
            Warna = float(Warna)
            
            # Scale numeric features
            scaled_features = scaler.transform([[pH, Temperatur]])
            
            # Combine all features into one array
            features = np.array([
                scaled_features[0][0], scaled_features[0][1],
                Rasa, Bau, Lemak, Kekeruhan, Warna
            ]).reshape(1, -1)
            
            # Make prediction with the model
            Prediksi_Susu = milkquality.predict(features)
            
            # Interpret prediction results
            if Prediksi_Susu[0] == 0:
                Prediksi_Susu = "high"
            elif Prediksi_Susu[0] == 1:
                Prediksi_Susu = "low"
            elif Prediksi_Susu[0] == 2:
                Prediksi_Susu = "medium"
            else:
                Prediksi_Susu = "unknown milk quality"
            
            st.success(Prediksi_Susu)
        else:
            st.error("Please fill in all the input fields.")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
