import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm

df = pd.read_csv('your_data.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(by='timestamp')

plt.figure(figsize=(10, 6))
plt.plot(df['timestamp'], df['sensor_value'], label='Original Data')
plt.title('Original Time-Series Data')
plt.xlabel('Timestamp')
plt.ylabel('Sensor Value')
plt.legend()
plt.show()

model = hmm.GaussianHMM(n_components=1, covariance_type="full")
model.fit(df[['sensor_value']])

hidden_states = model.predict(df[['sensor_value']])

plt.figure(figsize=(10, 6))
plt.plot(df['timestamp'], hidden_states, label='Hidden States')
plt.title('Hidden States Prediction')
plt.xlabel('Timestamp')
plt.ylabel('Hidden State')
plt.legend()
plt.show()

df['RUL'] = df.groupby('unit_number')['timestamp'].transform(lambda x: max(x) - x)
df['RUL'] = df['RUL'].dt.days

plt.figure(figsize=(10, 6))
plt.plot(df['timestamp'], df['RUL'], label='Remaining Useful Life')
plt.title('Remaining Useful Life Prediction')
plt.xlabel('Timestamp')
plt.ylabel('Remaining Useful Life (Days)')
plt.legend()
plt.show()
