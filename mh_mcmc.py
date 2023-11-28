import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Define the target distribution (the distribution you want to sample from)
# In this example, we'll use a simple univariate Gaussian distribution.
target_distribution = stats.norm(loc=0, scale=1)

# Define the number of iterations and initial state
num_iterations = 10000
initial_state = 0


# Initialize variables to store the samples and the acceptance count
samples = [initial_state]
acceptance_count = 0
eps = 0.25

# Metropolis-Hastings algorithm
for i in range(num_iterations):
    current_state = samples[-1]

    # Generate a proposed state from the proposal distribution
    proposed_state = stats.uniform.rvs(loc=current_state-eps, scale= 2*eps)
    #print(proposed_state)
    # Calculate the acceptance ratio
    acceptance_ratio = target_distribution.pdf(proposed_state) / target_distribution.pdf(current_state)
    
    # Accept or reject the proposed state based on the acceptance ratio
    if acceptance_ratio >= 1 or np.random.rand() < acceptance_ratio:
        samples.append(proposed_state)
        acceptance_count += 1
    else:
        samples.append(current_state)

# Calculate the acceptance rate
acceptance_rate = acceptance_count / num_iterations

print(f"Acceptance rate: {acceptance_rate}")



# Assuming you have a list of samples from your MCMC chain, named 'samples'
# samples = [...]

# Trace Plot
plt.figure(figsize=(12, 6))
plt.plot(samples, label='Trace')
plt.xlabel('Iteration')
plt.ylabel('Sample Value')
plt.title('Trace Plot')
plt.legend()
plt.grid(True)
plt.show()

# Autocorrelation Plot
def autocorrelation(samples, lag):
    N = len(samples)
    mean = np.mean(samples)
    numerator = np.sum((samples[:N - lag] - mean) * (samples[lag:] - mean))
    denominator = np.sum((samples - mean) ** 2)
    return numerator / denominator

lags = range(1, min(100, len(samples)))  # Adjust the maximum lag as needed
autocorr_values = [autocorrelation(samples, lag) for lag in lags]

plt.figure(figsize=(12, 6))
plt.plot(lags, autocorr_values, marker='o', linestyle='-', color='b', label='Autocorrelation')
plt.axhline(0, color='r', linestyle='--', label='Zero Autocorrelation')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation Plot')
plt.legend()
plt.grid(True)
plt.show()

# Define the thinning factor
thinning_factor = 10

# Thinning the samples
thinned_samples = samples[::thinning_factor]

# Create a histogram of the thinned samples
plt.figure(figsize=(12, 6))
plt.hist(thinned_samples, bins=30, density=True, alpha=0.7, color='blue', label='Sample Histogram')

# Add the PDF of a standard normal distribution for comparison
x = np.linspace(-3, 3, 1000)  # Adjust the range as needed
plt.plot(x, stats.norm.pdf(x, loc=0, scale=1), color='red', label='Standard Normal PDF')

plt.xlabel('Sample Value')
plt.ylabel('Density')
plt.title('Sample Histogram and Standard Normal PDF (Thinning Factor = 10)')
plt.legend()
plt.grid(True)
plt.show()