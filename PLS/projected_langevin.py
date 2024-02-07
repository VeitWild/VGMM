import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt

# Simulated input data X
torch.manual_seed(42)
num_points = 200
sigma = 0.1
input_dim = 1
X = torch.randn(num_points)  # Simulated 1D input data
Y = X**2 + 0.1*torch.randn(num_points)


#Inducing Points
Z = X 

# Length scale parameter and smoothness parameter for Matérn kernel
length_scale = 0.1
nu = 1.5  # Smoothness parameter (adjust as needed)


#Initiliase
nr_particles = 100
step_size = 0.01
t_end = 1000

U0  = torch.randn(input_dim,nr_particles)
U_old = U0

for t in range(0,t_end):
    Y_hat = 
    U_new = U_old +  1/sigma**2 *step_size * ( Y - Y_hat) 

    U_old = U_new



# Initializing Matérn kernel
matern_kernel = gpytorch.kernels.MaternKernel(nu=nu, ard_num_dims=1)
matern_kernel.lengthscale = length_scale  # Setting the length scale

# Building the kernel matrix for the input data X
with gpytorch.settings.fast_computations(covar_root_decomposition=True):
    # Enable fast computations for the kernel matrix (Cholesky root decomposition)
    kernel_matrix = matern_kernel(X)

print("Simulated Input Data X:")
print(X)

print("\nKernel Matrix:")
print(kernel_matrix.evaluate())

# Plotting the time series
plt.figure(figsize=(10, 6))
plt.scatter(X, Y)
plt.title("data plot")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()

