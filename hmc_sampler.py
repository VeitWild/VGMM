import numpy as np
from scipy.stats import norm

# Define the target distribution log likelihood function
def log_likelihood(x):
    return -0.5 * x**2

# Define the gradient of the log likelihood function
def grad_log_likelihood(x):
    return -x

# Leapfrog integration step
def leapfrog(x, r, step_size, num_steps, grad_log_likelihood):
    grad_noise = np.random.normal(scale=5)
    grad_log_likelihood_at_x = grad_log_likelihood(x)+ grad_noise

    r += 0.5 * step_size * grad_log_likelihood_at_x
    for _ in range(num_steps):
        x += step_size * r
        r += step_size * grad_log_likelihood_at_x
    r += 0.5 * step_size * grad_log_likelihood_at_x
    return x, r

# Hamiltonian Monte Carlo function
def hmc(log_likelihood, grad_log_likelihood, num_samples, num_steps, step_size):
    samples = []
    x = np.random.randn()
    
    for _ in range(num_samples):
        r = np.random.randn()  # Initialize momentum
        noise = np.random.normal(scale=5)
        current_U = -log_likelihood(x) + noise # Current potential energy
        current_K = 0.5 * np.sum(r**2)  # Current kinetic energy
        
        # Leapfrog integration
        x_new, r_new = leapfrog(x, r, step_size, num_steps, grad_log_likelihood)
        
        proposed_U = -log_likelihood(x_new)  # Proposed potential energy
        proposed_K = 0.5 * np.sum(r_new**2)  # Proposed kinetic energy
        
        # Metropolis acceptance criterion
        alpha = min(1, np.exp(current_U - proposed_U + current_K - proposed_K))
        if np.random.rand() < alpha:
            x = x_new
        samples.append(x)
    
    return np.array(samples)

# Parameters
num_samples = 10000
num_steps = 10
step_size = 0.1

# Run HMC
samples = hmc(log_likelihood, grad_log_likelihood, num_samples, num_steps, step_size)

# Display results
print("Mean:", np.mean(samples))
print("Standard Deviation:", np.std(samples))

from scipy.stats import norm
import matplotlib.pyplot as plt


# Display results
print("Mean:", np.mean(samples))
print("Standard Deviation:", np.std(samples))

# Plot histogram of HMC samples and standard normal distribution
plt.hist(samples, bins=50, density=True, alpha=0.5, color='blue', label='HMC Samples')
x = np.linspace(-5, 5, 100)
plt.plot(x, norm.pdf(x), 'r-', lw=2, label='Standard Normal PDF')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.title('HMC Samples vs. Standard Normal Distribution')
plt.show()
