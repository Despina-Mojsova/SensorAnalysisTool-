# Import necessary libraries
import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Generate normal sensor data (mean = 50, std = 5)
n_samples = 1000
normal_data = np.random.normal(loc=50, scale=5, size=n_samples)

# Generate anomalous data (sensor spikes in the higher range)
n_anomalies = 20
anomalies = np.random.uniform(low=70, high=100, size=n_anomalies)

# Combine normal and anomalous data into one signal
data = np.concatenate([normal_data, anomalies])

# Add random Gaussian noise and inject artificial spikes
noise = np.random.normal(scale=0.5, size=len(data))  # Low amplitude noise
spike_indices = np.random.choice(len(data), size=10, replace=False)
spikes = np.zeros_like(data)
spikes[spike_indices] = np.random.uniform(-3, 3, size=10)  # Sharp anomalies

# Final signal with data, noise, and spikes combined
final_signal = data + noise + spikes

# Create time stamps assuming 10 Hz sampling rate
time_stamps = np.arange(len(final_signal)) / 10

# Create DataFrame and export to CSV
sensor_df = pd.DataFrame({'Time': time_stamps, 'Sensor Value': final_signal})
sensor_df.to_csv("sensor_data.csv", index=False)

# Confirm file generation
print("sensor_data.csv")
