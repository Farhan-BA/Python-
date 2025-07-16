import streamlit as st
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load("glass_classifier_model.pkl")
scaler = joblib.load("glass_scaler.pkl")

# Streamlit app title and description
st.title("🔍 Glass Type Classifier")
st.markdown("This app predicts the **type of glass** based on its chemical composition using a trained ML model.")

# Input fields for chemical composition
st.sidebar.header("Enter Glass Composition")
RI = st.sidebar.number_input("Refractive Index (RI)", value=1.52)
Na = st.sidebar.number_input("Sodium (Na)", value=13.0)
Mg = st.sidebar.number_input("Magnesium (Mg)", value=2.0)
Al = st.sidebar.number_input("Aluminum (Al)", value=1.0)
Si = st.sidebar.number_input("Silicon (Si)", value=72.0)
K = st.sidebar.number_input("Potassium (K)", value=0.5)
Ca = st.sidebar.number_input("Calcium (Ca)", value=8.5)
Ba = st.sidebar.number_input("Barium (Ba)", value=0.0)
Fe = st.sidebar.number_input("Iron (Fe)", value=0.0)

# Collect input values into a numpy array
input_data = np.array([[RI, Na, Mg, Al, Si, K, Ca, Ba, Fe]])

# Scale the input data using the loaded scaler
input_scaled = scaler.transform(input_data)

# Make a prediction
prediction = model.predict(input_scaled)[0]

# Define glass types
glass_types = {
    1: "Building Windows (Float Process)",
    2: "Building Windows (Non Float Process)",
    3: "Vehicle Windows",
    4: "Containers",
    5: "Tableware",
    6: "Headlamps"
}

# Display the prediction
result = glass_types.get(prediction, "Unknown")
st.success(f"Predicted Glass Type: **{result} (Type {prediction})**")
