import pandas as pd
import numpy as np

# Settings
num_units = 5             # Number of engines
time_steps = 30           # Time steps per engine
num_features = 18         # Number of sensors

# For reproducibility
np.random.seed(9)

# Data generation
rows = []
for unit_id in range(1, num_units + 1):
    data = np.random.rand(time_steps, num_features)
    unit_ids = np.full((time_steps, 1), unit_id)
    full_data = np.hstack([unit_ids, data])
    rows.append(full_data)

# Combine all units
all_data = np.vstack(rows)

# Create DataFrame
columns = ['unit_id'] + [f"sensor_{i+1}" for i in range(num_features)]
df = pd.DataFrame(all_data, columns=columns)

# Save
df.to_csv("sample_batch_sensor_data.csv", index=False)

print("Batch sample CSV generated: sample_batch_sensor_data.csv")
