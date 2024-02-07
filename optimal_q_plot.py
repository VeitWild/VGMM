import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi

# Define a polynomial function of degree 4
def loss(x):
    return 3*(x + 1) ** 2 * (x - 1) ** 2

def refrence_pdf(x):
    return -0.5* x**2

def potential(x,reg_para=1):
    return loss(x)-reg_para*refrence_pdf(x)


# Generate x values
a =-1.5
b = 1.5
x = np.linspace(a, b, 400)  # Adjust the range as needed

# Calculate corresponding y values using the function
y = loss(x)

# Calculate y values for exp(-f(x))
Z, _ = spi.quad(potential, a, b)
y_exp = 1/Z * np.exp(-potential(x))

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot the original function in the first subplot
ax1.plot(x, y, label='loss')
ax1.set_xlabel(r'$\theta$')
ax1.set_ylabel(r'$\ell(\theta)$')
ax1.set_title('loss')
#ax1.axvline(x=0, color='red', linestyle='--', label='Change Point at x=0')
#ax1.scatter([-1, 1], [1, 1], color='green', label='Local Minima at x=-1 and x=1')
#ax1.legend()
ax1.grid(True)

# Plot the exp(-f(x)) in the second subplot
ax2.plot(x, y_exp, label=r'$q^*$')
ax2.set_xlabel(r'$\theta$')
ax2.set_ylabel(r'$q^*(\theta)$')
ax2.set_title(r'density of optimal $Q^*$')
#ax2.axvline(x=0, color='red', linestyle='--', label='Change Point at x=0')
#ax2.legend()
ax2.grid(True)

# Adjust layout for better visualization
plt.tight_layout()

# Show the plots
plt.show()
