import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Define a simple mixture distribution (e.g., a mixture of two Gaussian distributions)
def mixture_distribution(sample_size, num_components, means, stds, weights):
    component_indices = torch.multinomial(torch.tensor(weights), sample_size, replacement=True)
    samples = torch.empty(sample_size, len(means[0]))
    for i in range(sample_size):
        component_idx = component_indices[i].item()
        samples[i] = torch.normal(means[component_idx], stds[component_idx])
    return samples

# Simulate data from the mixture distribution
num_samples = 1000
num_components = 2
means = [torch.tensor([1.0, 1.0]), torch.tensor([-1.0, -1.0])]
stds = [0.1, 0.2]
weights = [0.7, 0.3]

simulated_data = mixture_distribution(num_samples, num_components, means, stds, weights)

# Create a target dataset
target_dataset = torch.utils.data.TensorDataset(simulated_data)

# Define the Diffusion Model for sampling
class DiffusionSampler(nn.Module):
    def __init__(self, num_steps, noise_schedule):
        super(DiffusionSampler, self).__init__()
        self.num_steps = num_steps
        self.noise_schedule = noise_schedule

        self.net = nn.Sequential(
            # Define your neural network architecture here
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
            # Add more layers as needed
        )

    def forward(self, x, t):
        for step in range(self.num_steps):
            noise = torch.randn_like(x) * (self.noise_schedule[step] ** 0.5)
            x += noise
            x = x + self.net(x) * (1.0 - self.noise_schedule[step])
        return x

# Define a function to generate noise schedule
def get_noise_schedule(num_steps):
    alphas = torch.linspace(0.0, 1.0, num_steps)
    return 1.0 - alphas

# Train the Diffusion Model to sample from the mixture distribution
def train_diffusion_sampler(diffusion_sampler, num_steps, num_epochs, batch_size, learning_rate, target_dataset):
    optimizer = optim.Adam(diffusion_sampler.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    dataloader = torch.utils.data.DataLoader(target_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        for data in dataloader:
            inputs = data[0]
            t = torch.rand(batch_size, 1)  # Sample t from a uniform distribution

            optimizer.zero_grad()
            outputs = diffusion_sampler(inputs, t)
            loss = loss_fn(outputs, inputs)
            loss.backward()
            optimizer.step()

            print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item()}')

if __name__ == "__main__":
    num_steps = 100  # Number of diffusion steps
    num_epochs = 10  # Number of training epochs
    batch_size = 64
    learning_rate = 1e-3

    noise_schedule = get_noise_schedule(num_steps)
    diffusion_sampler = DiffusionSampler(num_steps, noise_schedule)

    losses = train_diffusion_sampler(diffusion_sampler, num_steps, num_epochs, batch_size, learning_rate, target_dataset)

    # Generate samples from the trained diffusion sampler
    num_generated_samples = 1000
    generated_samples = torch.zeros(num_generated_samples, 2)

    for i in range(num_generated_samples):
        t = torch.rand(1, 1)  # Sample t from a uniform distribution
        z = torch.randn(1, 2)  # Sample random noise
        sample = diffusion_sampler(z, t)
        generated_samples[i] = sample

    # Plot the generated samples and the PDF of the mixture distribution
    plt.figure(figsize=(8, 6))
    plt.title("Generated Samples vs. Mixture Distribution PDF")

    # Plot the generated samples
    generated_samples_np = generated_samples.detach().numpy()  # Use .detach() to remove gradients
    plt.scatter(generated_samples_np[:, 0], generated_samples_np[:, 1], label="Generated Samples", alpha=0.7)
    #plt.scatter(generated_samples[:, 0], generated_samples[:, 1], label="Generated Samples", alpha=0.7)

   