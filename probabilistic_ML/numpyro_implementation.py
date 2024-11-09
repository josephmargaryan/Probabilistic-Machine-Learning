import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import numpy as np


class BayesianNeuralNetworkMCMC:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.circulant_matrix = np.random.randn(input_size, input_size)

    def model(self, x, y=None):
        # Define priors for weights and biases using Normal distributions
        fc1_weight = numpyro.sample(
            "fc1_weight",
            dist.Normal(
                jnp.zeros((self.hidden_size, self.input_size)),
                jnp.ones((self.hidden_size, self.input_size)),
            ),
        )
        fc1_bias = numpyro.sample(
            "fc1_bias",
            dist.Normal(jnp.zeros(self.hidden_size), jnp.ones(self.hidden_size)),
        )
        fc2_weight = numpyro.sample(
            "fc2_weight",
            dist.Normal(
                jnp.zeros((self.output_size, self.hidden_size)),
                jnp.ones((self.output_size, self.hidden_size)),
            ),
        )
        fc2_bias = numpyro.sample(
            "fc2_bias",
            dist.Normal(jnp.zeros(self.output_size), jnp.ones(self.output_size)),
        )

        # Forward pass using sampled weights and biases
        hidden = jnp.dot(x, fc1_weight.T) + fc1_bias
        hidden = jnp.maximum(hidden, 0)  # ReLU activation
        output = jnp.dot(hidden, fc2_weight.T) + fc2_bias

        # Define likelihood
        numpyro.sample("obs", dist.Normal(output, 0.1), obs=y)

    def forward_fft(self, x):
        # Use FFT for circulant matrix multiplication
        fft_x = np.fft.fft(x)
        circulant_fft = np.fft.fft(self.circulant_matrix)
        out = np.fft.ifft(fft_x * circulant_fft).real
        return out


# Running MCMC with NUTS


def run_mcmc(x_train, y_train):
    bnn = BayesianNeuralNetworkMCMC(
        input_size=x_train.shape[1], hidden_size=20, output_size=1
    )
    nuts_kernel = NUTS(bnn.model)
    mcmc = MCMC(nuts_kernel, num_samples=500, num_warmup=100)
    mcmc.run(random.PRNGKey(0), x_train, y_train)
    samples = mcmc.get_samples()
    return samples


# Uncertainty visualization function
def plot_uncertainty(x_train, y_train, x_test, samples):
    predictions = []
    for i in range(100):  # Sample 100 different parameter sets
        fc1_weight = samples["fc1_weight"][i]
        fc1_bias = samples["fc1_bias"][i]
        fc2_weight = samples["fc2_weight"][i]
        fc2_bias = samples["fc2_bias"][i]

        hidden = jnp.dot(x_test, fc1_weight.T) + fc1_bias
        hidden = jnp.maximum(hidden, 0)  # ReLU activation
        output = jnp.dot(hidden, fc2_weight.T) + fc2_bias
        predictions.append(output)

    predictions = np.array(predictions)

    # Ensure predictions are 2D for proper handling
    if predictions.ndim == 1:
        predictions = predictions[:, np.newaxis]

    # Reshape to (num_samples, num_data_points) for correct mean and std calculation
    predictions = predictions.reshape(predictions.shape[0], -1)

    mean_prediction = predictions.mean(axis=0)
    std_prediction = predictions.std(axis=0)

    plt.figure(figsize=(10, 6))
    plt.fill_between(
        np.arange(len(mean_prediction)),
        mean_prediction - std_prediction,
        mean_prediction + std_prediction,
        color="lightblue",
        alpha=0.5,
        label="Uncertainty",
    )
    plt.plot(mean_prediction, label="Mean Prediction", color="blue")
    plt.scatter(np.arange(len(y_train)), y_train, color="red", label="True Data")
    plt.title("Posterior Predictive Distribution with Uncertainty")
    plt.xlabel("Data Points")
    plt.ylabel("Predicted Value")
    plt.legend()
    plt.show()


def test_bayesian_neural_network_mcmc():
    # Generate synthetic data
    x_train = np.random.randn(100, 10)
    y_train = np.sin(x_train.sum(axis=1)) + 0.1 * np.random.randn(
        100
    )  # Noisy sine data

    # Run MCMC
    print("\nRunning MCMC with NUTS...")
    samples = run_mcmc(x_train, y_train)
    print("MCMC samples:", samples)

    # Analyze sample statistics (mean of weights as an example)
    for param_name, param_samples in samples.items():
        print(f"Mean of {param_name}: {param_samples.mean()}")

    # Generate test data for predictions
    x_test = np.random.randn(50, 10)
    plot_uncertainty(x_train, y_train, x_test, samples)


test_bayesian_neural_network_mcmc()
