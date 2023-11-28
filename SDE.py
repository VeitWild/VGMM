import numpy as np
import matplotlib.pyplot as plt

# Define the SDE parameters
mu = 0.1  # Drift term
sigma = 0.2  # Diffusion term
T = 1.0  # Total time
N = 1000  # Number of time steps
dt = T / N  # Time step size

# Initialize arrays to store time and the solution
t = np.linspace(0.0, T, N+1)
W = np.zeros(N+1)  # Brownian motion increments
X = np.ones(N+1)  # Solution of the SDE


# Generate Brownian motion using the increments

# Simulate the SDE using the Euler-Maruyama method
for i in range(1, N+1):
    dX = mu * X[i-1] * dt + sigma * X[i-1] * np.random.normal()*np.sqrt(dt)
    X[i] = X[i-1] + dX

# Plot the SDE solution
print(X)
plt.figure(figsize=(10, 5))
plt.plot(t, X, label='SDE Solution')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
