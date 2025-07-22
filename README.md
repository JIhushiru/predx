# Predictive Maintenance with RUL Prediction 

This project is a complete machine learning pipeline for **Remaining Useful Life (RUL) prediction** using time series sensor data from jet engines (CMAPSS Dataset - FD001). It is designed to support predictive maintenance use cases in industrial settings.

## Key Features

- Sensor data preprocessing and cleaning  
- Remaining Useful Life (RUL) labeling and threshold-based fault labeling  
- Sliding window transformation for time series modeling  
- LSTM-based neural network for RUL regression  
- Model saving and batch inference  
- Interactive visual dashboard (via Streamlit)

---

## Project Structure

```
├── app/
│   ├── main.py
│   ├── schemas.py
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   │   └── train_FD001.txt
│   ├── processed_FD001.csv
│   ├── scaled_FD001.csv
│   ├── test_split_FD001.csv
│   ├── train_sequences_FD001.npz
│   └── train_split_FD001.csv
├── model/
│   └── model.keras
├── notebooks/
│   └── 01_preprocessing.ipynb
├── sample/
│   ├── generate_sample_batch.csv.py
│   └── generate_sample.csv.py
├── .gitignore
├── README.md
├── requirements.txt
```

---

## Dashboard Features

Built using **Streamlit** (soon to be React), the dashboard allows:

- Uploading batch sensor files
- Visualizing individual unit sensor trends
- Highlighting units at risk (low RUL)
- Viewing histograms of RUL predictions
- (Coming soon) Real-time alert simulations & log integration

---

## Model Summary

- **Model Type:** LSTM  
- **Input Shape:** (window size, number of sensors)  
- **Loss Function:** Mean Squared Error (MSE)  
- **Optimizer:** Adam  
- **Framework:** TensorFlow / Keras

---

## Getting Started

1. Clone the repo  
   ```
   git clone https://github.com/JIhushiru/predx
   cd predx
   ```
2. Install dependencies:  
   ```
   pip install -r requirements.txt
   ```
3. Start the server:
   ```
   uvicorn app.main:app --reload
   ```
4. Run the Streamlit app in another terminal (from inside the app directory): 
   ```
   cd app
   streamlit run streamlit_app.py
   ```

---

## Real-World Application

This project is built to showcase the **core tasks involved in predictive maintenance**, including:

- Degradation modeling  
- Failure forecasting  
- Intelligent diagnostics for asset management  

Perfect for roles in:
- Industrial IoT
- Reliability Engineering
- Machine Learning for Operations

---

## Dataset

Source: [NASA CMAPSS Data](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)

---

## Author

Built by Jer Heseoh R. Arsolon 

---
