from pydantic import BaseModel
from typing import List

class SensorInput(BaseModel):
    data: List[List[float]]  # shape: (sequence_length, num_features)
