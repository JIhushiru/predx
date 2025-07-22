from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import tensorflow as tf
from app.schemas import SensorInput

app = FastAPI()
model = tf.keras.models.load_model("model/model.keras")

@app.post("/predict_rul")
def predict_rul(input_data: SensorInput):
    # Convert to numpy and add batch dimension
    X = np.expand_dims(np.array(input_data.data), axis=0)  # (1, sequence_length, features)
    prediction = model.predict(X)
    predicted_rul = prediction[0][0]
    return {"predicted_RUL": float(predicted_rul)}
