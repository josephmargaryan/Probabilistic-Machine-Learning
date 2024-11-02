import numpy as np
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# 1. Circulant Matrix Multiplication using FFT
def circulant_multiply(w, x):
    """
    Multiply circulant matrix defined by vector w with vector x using FFT.
    """
    w_fft = fft(w)
    x_fft = fft(x)
    return np.real(ifft(w_fft * x_fft))

# 2. Bayesian Neural Network Model using Circulant Matrices
def circulant_nn_model(x, y=None):
    # Input dimension
    D = x.shape[1]
    
    # Prior for circulant weight vector
    w1 = numpyro.sample('w1', dist.Normal(jnp.zeros(D), jnp.ones(D)))
    b1 = numpyro.sample('b1', dist.Normal(0., 1.))
    
    # Hidden layer computation using circulant multiplication
    hidden = jnp.tanh(circulant_multiply(w1, x.T).T + b1)
    
    # Output layer weights (assuming scalar output)
    w2 = numpyro.sample('w2', dist.Normal(jnp.zeros(D), jnp.ones(D)))
    b2 = numpyro.sample('b2', dist.Normal(0., 1.))
    
    # Output computation
    logits = jnp.dot(hidden, w2) + b2
    # Likelihood
    numpyro.sample('obs', dist.Normal(logits, 0.1), obs=y)

# 3. Generate Synthetic Dataset
np.random.seed(0)
N = 100  # Number of data points
D = 16   # Input dimension (power of 2 for FFT efficiency)

X = np.random.randn(N, D)
true_w = np.random.randn(D)
y = np.dot(X, true_w) + np.random.randn(N) * 0.1

# Convert data to JAX arrays
X_jax = jnp.array(X)
y_jax = jnp.array(y)

# 4. Run MCMC Sampling
kernel = NUTS(circulant_nn_model)
mcmc = MCMC(kernel, num_warmup=500, num_samples=1000)
rng_key = random.PRNGKey(0)
mcmc.run(rng_key, x=X_jax, y=y_jax)

samples = mcmc.get_samples()

# 5. Extract Posterior Means
w1_posterior_mean = samples['w1'].mean(axis=0)
w2_posterior_mean = samples['w2'].mean(axis=0)
b1_posterior_mean = samples['b1'].mean()
b2_posterior_mean = samples['b2'].mean()

# 6. Predictions
def predict(x, w1, b1, w2, b2):
    hidden = jnp.tanh(circulant_multiply(w1, x.T).T + b1)
    logits = jnp.dot(hidden, w2) + b2
    return logits

y_pred = predict(X_jax, w1_posterior_mean, b1_posterior_mean, w2_posterior_mean, b2_posterior_mean)

# 7. Evaluate the Model
mse = mean_squared_error(y, y_pred)
print(f'Mean Squared Error: {mse}')

# 8. Visualizations

# 8.1 Posterior Mean and Variance of Weights (w1)
def plot_posterior_weights(samples):
    w1_mean = samples['w1'].mean(axis=0)
    w1_var = samples['w1'].var(axis=0)

    plt.figure(figsize=(12, 5))
    
    # Plot posterior mean of weights
    plt.subplot(1, 2, 1)
    plt.bar(range(len(w1_mean)), w1_mean)
    plt.title("Posterior Mean of w1")
    plt.xlabel("Weight Index")
    plt.ylabel("Mean")
    
    # Plot posterior variance of weights
    plt.subplot(1, 2, 2)
    plt.bar(range(len(w1_var)), w1_var)
    plt.title("Posterior Variance of w1")
    plt.xlabel("Weight Index")
    plt.ylabel("Variance")
    
    plt.tight_layout()
    plt.show()

# 8.2 Plot Predictions vs Actual
def plot_predictions_vs_actual(y_true, y_pred):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, c="b", label="Predicted")
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', lw=2, label="Ideal")
    plt.title("Predicted vs Actual Values")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.show()

# 8.3 Posterior Predictive Check
def posterior_predictive_check(y_true, y_pred):
    plt.figure(figsize=(8, 6))
    plt.hist(y_true - y_pred, bins=20, color='blue', alpha=0.7)
    plt.title("Posterior Predictive Check: Residuals")
    plt.xlabel("Residuals (y_true - y_pred)")
    plt.ylabel("Frequency")
    plt.show()

# Call visualization functions
plot_posterior_weights(samples)
plot_predictions_vs_actual(y, y_pred)
posterior_predictive_check(y, y_pred)


