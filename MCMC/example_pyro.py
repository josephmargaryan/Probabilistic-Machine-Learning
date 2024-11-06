import pyro
import pyro.distributions as dist
import torch

# Linear regression model in Pyro
def linear_regression_model(x, y=None):
    # Define priors for parameters a, b, and sigma
    a = pyro.sample("a", dist.Normal(0, 10))     # Intercept
    b = pyro.sample("b", dist.Normal(0, 10))     # Slope
    sigma = pyro.sample("sigma", dist.HalfCauchy(5.0))  # Noise standard deviation (positive)
    
    # Define the likelihood
    with pyro.plate("data", len(x)):
        y_hat = a + b * x
        pyro.sample("obs", dist.Normal(y_hat, sigma), obs=y)

# Generate synthetic data
torch.manual_seed(0)
x_data = torch.linspace(0, 10, 100)
true_a, true_b, true_sigma = 1.5, 2.5, 1.0
y_data = true_a + true_b * x_data + torch.randn(100) * true_sigma


from pyro.infer import MCMC, NUTS

# Set up the NUTS sampler and MCMC
nuts_kernel = NUTS(linear_regression_model)
mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=200)
mcmc.run(x_data, y_data)

# Extract posterior samples
posterior_samples = mcmc.get_samples()


import matplotlib.pyplot as plt

# Plot posterior distributions for a, b, and sigma
plt.figure(figsize=(10, 3))
for i, param in enumerate(['a', 'b', 'sigma']):
    plt.subplot(1, 3, i + 1)
    plt.hist(posterior_samples[param].numpy(), bins=30, density=True)
    plt.title(f"Posterior of {param}")
plt.show()
