from gp_regression import GaussianProcessRegression
import gpytorch
from sklearn.model_selection import train_test_split
import numpy as np

import itertools
from sklearn.metrics import mean_squared_error

class GPHyperparameterTuner:
    def __init__(self, gp_class, metric=mean_squared_error):
        """
        Initialize the hyperparameter tuner.

        Args:
            gp_class: The GaussianProcessRegression class.
            metric: The evaluation metric (default is mean squared error).
        """
        self.gp_class = gp_class
        self.metric = metric

    def tune(self, X_train, y_train, X_val, y_val, param_grid):
        """
        Perform hyperparameter tuning using grid search.

        Args:
            X_train, y_train: Training data.
            X_val, y_val: Validation data.
            param_grid: Dictionary of hyperparameter lists to search over.

        Returns:
            best_params: The best hyperparameter combination.
            best_score: The best validation score.
        """
        best_score = float('inf')
        best_params = None

        # Generate all combinations of hyperparameters
        keys, values = zip(*param_grid.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        # Iterate through all combinations
        for params in param_combinations:
            print(f"Testing parameters: {params}")
            
            # Create a new instance of the GP with the given parameters
            gp = self.gp_class(
                kernel=params.get("kernel"),
                noise=params.get("noise", 0.1)
            )

            # Train the model
            gp.train(X_train, y_train, num_epochs=params.get("num_epochs", 50), lr=params.get("lr", 0.1))
            
            # Predict on validation data
            mean, _ = gp.predict(X_val)
            
            # Compute the metric
            score = self.metric(y_val, mean)
            print(f"Validation Score (MSE): {score}")

            # Update the best score and parameters
            if score < best_score:
                best_score = score
                best_params = params

        return best_params, best_score


if __name__ == "__main__":
    param_grid = {
    "kernel": [
        gpytorch.kernels.RBFKernel(),
        gpytorch.kernels.MaternKernel(),
        gpytorch.kernels.RBFKernel() + gpytorch.kernels.LinearKernel()
    ],
    "noise": [0.01, 0.1, 1.0],
    "num_epochs": [50, 100],
    "lr": [0.01, 0.1]
    }

    from sklearn.model_selection import train_test_split
    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate synthetic data
    n_samples, n_features = 500, 5
    X = np.random.rand(n_samples, n_features) * 10  # Random features in range [0, 10]
    true_beta = np.array([3.5, -2.1, 0.8, 4.2, -1.5])  # True coefficients
    noise = np.random.normal(0, 2.0, size=n_samples)  # Gaussian noise

    # Linear relationship
    y = X @ true_beta + noise


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Instantiate the tuner
    tuner = GPHyperparameterTuner(gp_class=GaussianProcessRegression)

    # Run the hyperparameter tuning
    best_params, best_score = tuner.tune(X_train, y_train, X_test, y_test, param_grid)

    print("\nBest Parameters:", best_params)
    print("Best Validation Score:", best_score)
