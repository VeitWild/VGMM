import numpy as np
import matplotlib.pyplot as plt

#np.random.seed(42)  # Setting seed for reproducibility

# Function to generate the time series
def generate_time_series(num_points):
    Y = np.zeros(num_points)
    Y[0] = np.random.uniform(-2, 2)  # Initial value for Y

    for t in range(1, num_points):
        noise = np.random.normal(0, 0.25)  # Generating Gaussian noise
        Y[t] = np.sin(Y[t - 1]**2) + noise  # Generating the time series based on the rule

    return Y

# Generating the time series data
num_points = 2000
time_series = generate_time_series(num_points)

# Plotting the time series
plt.figure(figsize=(10, 6))
plt.plot(np.arange(num_points), time_series, marker='o', linestyle='-')
plt.title("Generated Time Series: Y_{t+1} = sin(Y_t) + noise")
plt.xlabel("Time")
plt.ylabel("Y")
plt.grid(True)
plt.show()
