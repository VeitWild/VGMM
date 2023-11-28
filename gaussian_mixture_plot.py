import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Define parameters for the mixture components
mean1 = np.array([-2, 2])
mean2 = np.array([2, 2])
mean3 = np.array([-2, -2])
mean4 = np.array([2, -2])

cov1 = np.array([[0.1, 0], [0, 0.1]])
cov2 = cov1
cov3 = cov1
cov4 = cov1

#cov2 = np.array([[0.5, -0.3], [-0.3, 0.5]])
#cov3 = np.array([[0.5, 0.3], [0.3, 0.5]])
#cov4 = np.array([[0.5, -0.3], [-0.3, 0.5]])

weights = [0.25, 0.25, 0.25, 0.25]

# Number of samples to generate
n_samples = 400

# Generate samples from the mixture distribution
samples = []
for _ in range(n_samples):
    component = np.random.choice(4, p=weights)
    if component == 0:
        sample = np.random.multivariate_normal(mean1, cov1)
    elif component == 1:
        sample = np.random.multivariate_normal(mean2, cov2)
    elif component == 2:
        sample = np.random.multivariate_normal(mean3, cov3)
    else:
        sample = np.random.multivariate_normal(mean4, cov4)
    samples.append(sample)

samples = np.array(samples)
samples = np.random.multivariate_normal(mean1,cov1,size=n_samples)


# Plot the samples
plt.figure(figsize=(8, 6))
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5, label='Samples',color='black')

# Define a grid to evaluate the log density
x, y = np.meshgrid(np.linspace(-5, 5, 400), np.linspace(-5, 5, 400))
pos = np.dstack((x, y))

# Calculate the log density for each component and sum them
log_density = 0
for i, (mean, cov) in enumerate(zip([mean1, mean2, mean3, mean4], [cov1, cov2, cov3, cov4])):
    component_density = multivariate_normal(mean=mean, cov=cov)
    log_density += weights[i] * component_density.logpdf(pos)

density = 0
means = [mean1,mean2,mean3,mean4]
covs = [cov1,cov2,cov3,cov4]
for i in range(0,4):
    component_density = multivariate_normal(mean=means[i], cov=covs[i])
    density += weights[i] * component_density.pdf(pos)





# Plot the log density
plt.contourf(x, y, np.log(density), levels=25, cmap='viridis', alpha=0.5)
plt.colorbar()
plt.title('Finite Dimensional/Parameterised GVI')
plt.xlabel(r'$\theta_1$')
plt.ylabel(r'$\theta_2$')
plt.legend()
plt.show()
