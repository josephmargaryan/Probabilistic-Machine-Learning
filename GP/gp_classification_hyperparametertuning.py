from gp_classification import GaussianProcessClassification
import gpytorch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
import itertools
from sklearn.metrics import accuracy_score

class GPHyperparameterTunerClassification:
    def __init__(self, gp_class, metric=accuracy_score):
        """
        Initialize the hyperparameter tuner.

        Args:
            gp_class: The GaussianProcessClassification class.
            metric: The evaluation metric (default is accuracy score).
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
        best_score = -float('inf')  # For accuracy, higher is better
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
                num_inducing_points=params.get("num_inducing_points", 100)
            )

            # Train the model
            gp.train(X_train, y_train, num_epochs=params.get("num_epochs", 50), lr=params.get("lr", 0.1))

            # Predict on validation data
            probs = gp.predict(X_val)
            preds = (probs > 0.5).astype(int)  # Convert probabilities to binary predictions

            # Compute the metric
            score = self.metric(y_val, preds)
            print(f"Validation Score (Accuracy): {score:.4f}")

            # Update the best score and parameters
            if score > best_score:
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
    "num_inducing_points": [50, 100],
    "num_epochs": [50, 100],
    "lr": [0.01, 0.1]
    }

    # Generate synthetic classification data
    X, y = make_classification(n_samples=500, n_features=5, n_informative=3, n_classes=2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Instantiate the tuner
    tuner = GPHyperparameterTunerClassification(gp_class=GaussianProcessClassification)

    # Run the hyperparameter tuning
    best_params, best_score = tuner.tune(X_train, y_train, X_val, y_val, param_grid)

    print("\nBest Parameters:", best_params)
    print("Best Validation Accuracy:", best_score)



