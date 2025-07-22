import streamlit as st
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.title("RUL Predictor")

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

@st.cache_data
def predict_unit(features_list):
    try:
        response = requests.post("http://localhost:8000/predict_rul", json={"data": features_list})
        result = response.json()
        return result.get("predicted_RUL", None)
    except:
        return None

uploaded_file = st.file_uploader("Upload sensor data CSV")
if uploaded_file:
    df = load_data(uploaded_file)

    tab1, tab2 = st.tabs(["📦 Batch Prediction", "🔍 Single Unit Prediction"])

    with tab1:
        if 'unit_id' in df.columns:
            unit_ids = df['unit_id'].unique()
            st.write(f"Detected {len(unit_ids)} units.")
            batch_results = []

            for unit in unit_ids:
                unit_df = df[df['unit_id'] == unit].drop(columns=['unit_id'])
                if unit_df.shape[0] >= 30 and unit_df.shape[1] == 18:
                    features = unit_df.values[-30:].tolist()
                    predicted_rul = predict_unit(features)
                    if predicted_rul is not None:
                        batch_results.append((unit, predicted_rul))
                else:
                    st.warning(f"Skipping unit {unit}: insufficient rows or wrong number of features.")

            if batch_results:
                st.subheader("Batch Predictions")
                result_df = pd.DataFrame(batch_results, columns=["Unit ID", "Predicted RUL"])
                st.dataframe(result_df)

                selected_unit = st.selectbox("Inspect Sensor Trend for Unit", [unit for unit, _ in batch_results])
                unit_df = df[df['unit_id'] == selected_unit].drop(columns=['unit_id'])
                last_30 = unit_df.values[-30:]

                st.subheader(f"Sensor Trend for Unit {selected_unit}")
                fig, ax = plt.subplots(figsize=(12, 6))
                for i in range(18):
                    ax.plot(range(30), last_30[:, i], label=f"Sensor {i+1}")
                ax.set_xlabel("Time Step")
                ax.set_ylabel("Sensor Reading")
                ax.set_title(f"Sensor Trends - Unit {selected_unit}")
                ax.legend(ncol=3, fontsize=8)
                st.pyplot(fig)

                threshold = st.slider("RUL Threshold for At-Risk Units", min_value=0, max_value=100, value=20)
                at_risk_units = [unit for unit, rul in batch_results if rul < threshold]
                if at_risk_units:
                    st.warning(f"⚠️ Units below RUL threshold ({threshold}): {', '.join(map(str, at_risk_units))}")

                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Batch Predictions as CSV", data=csv, file_name='rul_predictions.csv', mime='text/csv')
        else:
            st.info("This tab is for datasets with a `unit_id` column.")

    with tab2:
        if 'unit_id' not in df.columns:
            if df.shape[1] != 18:
                st.error(f"Expected 18 features, but got {df.shape[1]}")
            elif df.shape[0] < 30:
                st.error(f"Expected at least 30 rows (time steps), but got {df.shape[0]}")
            else:
                features = df.values[-30:].tolist()
                predicted_rul = predict_unit(features)

                if predicted_rul is not None:
                    st.success(f"Predicted RUL: {predicted_rul:.2f}")
                    single_df = pd.DataFrame([["001", predicted_rul]], columns=["Unit ID", "Predicted RUL"])
                    csv = single_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Prediction as CSV", data=csv, file_name='single_rul_prediction.csv', mime='text/csv')

                    if predicted_rul < 20:
                        st.warning("System approaching failure — consider maintenance soon.")

                    st.subheader("Sensor Data Trend (Last 30 Timesteps)")
                    fig, ax = plt.subplots(figsize=(12, 6))
                    last_30 = np.array(features)
                    for i in range(18):
                        ax.plot(range(30), last_30[:, i], label=f"Feature {i+1}")
                    ax.set_xlabel("Time Step")
                    ax.set_ylabel("Sensor Reading")
                    ax.set_title("Sensor Trends")
                    ax.legend(ncol=3, fontsize=8)
                    st.pyplot(fig)
                else:
                    st.error("Prediction failed.")
        else:
            st.info("This tab is for sensor files **without** a `unit_id` column.")
