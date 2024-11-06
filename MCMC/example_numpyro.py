import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import random

# Linear regression model in NumPyro
def linear_regression_model(x, y=None):
    a = numpyro.sample("a", dist.Normal(0, 10))
    b = numpyro.sample("b", dist.Normal(0, 10))
    sigma = numpyro.sample("sigma", dist.HalfCauchy(5.0))
    
    y_hat = a + b * x
    numpyro.sample("obs", dist.Normal(y_hat, sigma), obs=y)

# Generate synthetic data
np.random.seed(0)
x_data = np.linspace(0, 10, 100)
y_data = true_a + true_b * x_data + np.random.normal(0, true_sigma, size=100)


from numpyro.infer import MCMC, NUTS

# Set up the NUTS sampler and MCMC
rng_key = random.PRNGKey(0)
nuts_kernel = NUTS(linear_regression_model)
mcmc = MCMC(nuts_kernel, num_samples=1000, num_warmup=200)
mcmc.run(rng_key, x=jnp.array(x_data), y=jnp.array(y_data))

# Extract posterior samples
posterior_samples = mcmc.get_samples()


import seaborn as sns

# Plot posterior distributions for a, b, and sigma
plt.figure(figsize=(10, 3))
for i, param in enumerate(['a', 'b', 'sigma']):
    plt.subplot(1, 3, i + 1)
    sns.histplot(posterior_samples[param], kde=True, bins=30)
    plt.title(f"Posterior of {param}")
plt.show()


