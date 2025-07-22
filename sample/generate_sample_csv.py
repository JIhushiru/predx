import pandas as pd
import numpy as np

# Dimensions
num_rows = 30  # time steps
num_features = 18  # features per time step

# Generate random data (change seed for reproducibility)
np.random.seed(90)
data = np.random.rand(num_rows, num_features)

# Create a DataFrame
df = pd.DataFrame(data, columns=[f"sensor_{i+1}" for i in range(num_features)])

# Save to CSV
df.to_csv("sample_sensor_data.csv", index=False)

print("Sample CSV generated: sample_sensor_data.csv")
