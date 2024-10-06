import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from numpyro import sample, plate
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, HMCECS, autoguide, SVI, Trace_ELBO
from numpyro.optim import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


# 1. Circulant Matrix using JAX's FFT
def circulant_multiply(w, X):
    """Multiply circulant matrix W (defined by vector w) by X using FFT."""
    w_fft = jnp.fft.fft(w)  # (n_features,)
    X_fft = jnp.fft.fft(X, axis=1)  # (num_samples, n_features)
    result_fft = X_fft * w_fft  # Broadcasting over features
    result = jnp.fft.ifft(result_fft, axis=1).real  # (num_samples, n_features)
    return result  # (num_samples, n_features)


# 2. Define the Bayesian Deep Learning Model
def bayesian_model(X, Y=None, subsample_size=None):
    n_samples, n_features = X.shape

    # Prior on the weights (using circulant matrix structure)
    w = sample("w", dist.Normal(jnp.zeros(n_features), 1.0))
    b = sample("b", dist.Normal(0.0, 1.0))

    with plate("data", size=n_samples, subsample_size=subsample_size) as idx:
        # Subsample data using indices
        X_batch = X[idx, :]  # Shape: (subsample_size, n_features)
        Y_batch = None if Y is None else Y[idx]  # Shape: (subsample_size,)
        # Compute logits
        logits_batch = (
            circulant_multiply(w, X_batch).sum(axis=1) + b
        )  # Shape: (subsample_size,)
        sample("obs", dist.Bernoulli(logits=logits_batch), obs=Y_batch)


# 3. Define HMC-ECS for Efficient Subsampling MCMC
def run_hmcecs(
    rng_key, X, Y, subsample_size, num_warmup=500, num_samples=1000, num_blocks=100
):
    # Define the NUTS kernel as the inner kernel
    kernel = NUTS(bayesian_model)

    # Subsampling proxy (Energy Conserving Subsampling)
    init_params = {"w": jnp.zeros(X.shape[1]), "b": 0.0}
    proxy = HMCECS.taylor_proxy(init_params)

    # HMCECS Kernel
    hmcecs_kernel = HMCECS(kernel, num_blocks=num_blocks, proxy=proxy)

    # Run the MCMC with subsampling
    mcmc = MCMC(hmcecs_kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(rng_key, X, Y, subsample_size=subsample_size)
    mcmc.print_summary()
    return mcmc.get_samples()


# 4. SVI as Initialization for MCMC
def initialize_with_svi(rng_key, X, Y, subsample_size, num_steps=2000):
    guide = autoguide.AutoDelta(bayesian_model)
    svi = SVI(bayesian_model, guide, Adam(0.005), loss=Trace_ELBO())
    svi_result = svi.run(rng_key, num_steps, X, Y, subsample_size=subsample_size)
    return svi_result.params


# 5. Visualization functions
def plot_posterior_samples(mcmc_samples):
    theta_mean = jnp.mean(mcmc_samples["w"], axis=0)
    theta_var = jnp.var(mcmc_samples["w"], axis=0)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(jnp.sort(theta_mean), "o-r", label="Mean of weights")
    ax[0].set_title("Posterior Mean of Weights")
    ax[0].set_xlabel("Weight Index")
    ax[0].set_ylabel("Mean")

    ax[1].plot(jnp.sort(theta_var), "o-b", label="Variance of weights")
    ax[1].set_title("Posterior Variance of Weights")
    ax[1].set_xlabel("Weight Index")
    ax[1].set_ylabel("Variance")

    plt.tight_layout()
    plt.show()


def plot_predictions(mcmc_samples, X, Y):
    theta_mean = jnp.mean(mcmc_samples["w"], axis=0)
    b_mean = jnp.mean(mcmc_samples["b"], axis=0)

    logits = circulant_multiply(theta_mean, X).sum(axis=1) + b_mean
    predictions = (logits > 0).astype(int)  # Adjust threshold as needed

    # Plot actual vs predicted
    plt.figure(figsize=(8, 5))
    plt.scatter(jnp.arange(len(Y)), Y, color="r", label="Actual")
    plt.scatter(
        jnp.arange(len(Y)), predictions, color="b", label="Predicted", alpha=0.5
    )
    plt.title("Actual vs Predicted Labels")
    plt.xlabel("Sample Index")
    plt.ylabel("Label")
    plt.legend()
    plt.show()


# 6. Evaluation Function
def evaluate_model(mcmc_samples, X, Y):
    theta_mean = jnp.mean(mcmc_samples["w"], axis=0)
    b_mean = jnp.mean(mcmc_samples["b"], axis=0)

    # Compute logits and make predictions
    logits = circulant_multiply(theta_mean, X).sum(axis=1) + b_mean
    predictions = (logits > 0).astype(int)  # Adjust threshold as needed

    # Convert JAX arrays to NumPy arrays for sklearn metrics
    Y_np = np.array(Y)
    predictions_np = np.array(predictions)
    logits_np = np.array(logits)

    # Calculate evaluation metrics
    accuracy = accuracy_score(Y_np, predictions_np)
    precision = precision_score(Y_np, predictions_np)
    recall = recall_score(Y_np, predictions_np)
    f1 = f1_score(Y_np, predictions_np)
    auc = roc_auc_score(Y_np, logits_np)
    cm = confusion_matrix(Y_np, predictions_np)

    # Print out the metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    print("Confusion Matrix:")
    print(cm)


# 7. Main function to run the model
def main():
    # Set random seed
    rng_key = random.PRNGKey(0)
    rng_key, data_key = random.split(rng_key)

    # Generate data using JAX's random functions
    X = random.normal(data_key, shape=(200, 20))
    Y = (X.sum(axis=1) > 0).astype(int)

    # Initialize with SVI
    svi_params = initialize_with_svi(rng_key, X, Y, subsample_size=50)

    # Run HMC-ECS
    rng_key, mcmc_key = random.split(rng_key)
    mcmc_samples = run_hmcecs(mcmc_key, X, Y, subsample_size=50)

    # Plot posterior samples
    plot_posterior_samples(mcmc_samples)
    plot_predictions(mcmc_samples, X, Y)

    # Evaluate the model performance
    evaluate_model(mcmc_samples, X, Y)

    # Output MCMC results
    print("Posterior samples:", mcmc_samples)


if __name__ == "__main__":
    main()
