import numpy as np
import pickle

class GlassTypeModel:
    def __init__(self):
        with open("glass_classifier_model.pkl", "rb") as model_file:
            self.model = pickle.load(model_file)

        with open("glass_scaler.pkl", "rb") as scaler_file:
            self.scaler = pickle.load(scaler_file)

    def predict(self, input_data):
        """Input: list of 9 features, Output: predicted glass type"""
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = self.scaler.transform(input_array)
        prediction = self.model.predict(input_scaled)
        return int(prediction[0])

import streamlit as st
from glass_predictor import GlassTypeModel

st.set_page_config(page_title="Glass Type Classifier", layout="centered")
st.title("🔍 Glass Type Predictor")
st.write("Enter the chemical properties of a glass sample:")

features = ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]
input_data = []

for feature in features:
    val = st.number_input(f"{feature}:", step=0.01, format="%.4f")
    input_data.append(val)

if st.button("Predict"):
    model = GlassTypeModel()
    result = model.predict(input_data)
    st.success(f"✅ Predicted Glass Type: **{result}**")











