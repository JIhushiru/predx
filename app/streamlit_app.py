import streamlit as st
import requests
import numpy as np
import pandas as pd

st.title("RUL Predictor")

uploaded_file = st.file_uploader("Upload sensor data CSV")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    if df.shape[1] != 18:
        st.error(f"Expected 18 features, but got {df.shape[1]}")
    elif df.shape[0] < 30:
        st.error(f"Expected at least 30 rows (time steps), but got {df.shape[0]}")
    else:
        features = df.values[-30:].tolist()  # last 30 time steps
        try:
            response = requests.post("http://localhost:8000/predict_rul", json={"data": features})
            result = response.json()

            if response.status_code == 200 and "predicted_RUL" in result:
                st.success(f"Predicted RUL: {result['predicted_RUL']:.2f}")
            else:
                st.error(f"Backend error: {result}")
            if result["predicted_RUL"] < 20:
                st.warning("System approaching failure — consider maintenance soon.")
        except Exception as e:
            st.error(f"Request failed: {e}")
