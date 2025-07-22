import streamlit as st
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.title("RUL Predictor")

uploaded_file = st.file_uploader("Upload sensor data CSV")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'unit_id' in df.columns:
        unit_ids = df['unit_id'].unique()
        st.write(f"Detected {len(unit_ids)} units.")
        batch_results = []

        for unit in unit_ids:
            unit_df = df[df['unit_id'] == unit].drop(columns=['unit_id'])
            if unit_df.shape[0] >= 30 and unit_df.shape[1] == 18:
                features = unit_df.values[-30:].tolist()
                response = requests.post("http://localhost:8000/predict_rul", json={"data": features})
                result = response.json()
                if "predicted_RUL" in result:
                    batch_results.append((unit, result["predicted_RUL"]))
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

            # User-defined RUL risk threshold
            threshold = st.slider("RUL Threshold for At-Risk Units", min_value=0, max_value=100, value=20)

            # Highlight at-risk units based on threshold 
            at_risk_units = [unit for unit, rul in batch_results if rul < threshold]
            if at_risk_units:
                st.warning(f"⚠️ Units below RUL threshold ({threshold}): {', '.join(map(str, at_risk_units))}")

            # Download button
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Batch Predictions as CSV",
                data=csv,
                file_name='rul_predictions.csv',
                mime='text/csv',
            )

    else:
        if df.shape[1] != 18:
            st.error(f"Expected 18 features, but got {df.shape[1]}")
        elif df.shape[0] < 30:
            st.error(f"Expected at least 30 rows (time steps), but got {df.shape[0]}")
        else:
            features = df.values[-30:].tolist()
            try:
                response = requests.post("http://localhost:8000/predict_rul", json={"data": features})
                result = response.json()

                if response.status_code == 200 and "predicted_RUL" in result:
                    rul_value = result['predicted_RUL']
                    st.success(f"Predicted RUL: {rul_value:.2f}")

                    # Offer download for single prediction
                    single_df = pd.DataFrame([["001", rul_value]], columns=["Unit ID", "Predicted RUL"])
                    csv = single_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Prediction as CSV",
                        data=csv,
                        file_name='single_rul_prediction.csv',
                        mime='text/csv',
                    )


                    # Show warning if RUL is low
                    if result["predicted_RUL"] < 20:
                        st.warning("System approaching failure — consider maintenance soon.")

                    # Plot trend for each feature
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
                    st.error(f"Backend error: {result}")
            except Exception as e:
                st.error(f"Request failed: {e}")