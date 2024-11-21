from gp_regression import GaussianProcessRegression
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from sklearn.model_selection import train_test_split



def plot_loss_curve(loss_values):
    """
    Plots the loss curve for GPR training.

    Args:
        loss_values: List of loss values recorded during training.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(loss_values)), loss_values, label="NLL", color="blue")
    plt.title("Negative Log-Marginal Likelihood During Training")
    plt.xlabel("Epochs")
    plt.ylabel("NLL")
    plt.legend()
    plt.grid()
    plt.show()


def plot_confidence_intervals_per_feature(X_train, y_train, X_test, mean, std, feature_index=0):
    """
    Plots mean predictions and confidence intervals for a single feature.

    Args:
        X_train: Training features.
        y_train: Training targets.
        X_test: Test features.
        mean: Predicted mean values.
        std: Predicted standard deviations.
        feature_index: Index of the feature to plot (default is 0).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Use the selected feature for visualization
    train_feature = X_train[:, feature_index]
    test_feature = X_test[:, feature_index]

    plt.figure(figsize=(10, 6))
    plt.scatter(train_feature, y_train, color="blue", label="Training Data", alpha=0.6)
    plt.plot(test_feature, mean, color="red", label="Mean Prediction", linewidth=2)
    plt.fill_between(
        test_feature,
        mean - 1.96 * std,
        mean + 1.96 * std,
        color="pink",
        alpha=0.3,
        label="95% Confidence Interval"
    )
    plt.title(f"Gaussian Process Regression: Confidence Intervals for Feature {feature_index}")
    plt.xlabel(f"Feature {feature_index}")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.show()


def plot_residuals(y_true, y_pred):
    """
    Plots residuals of the predictions.

    Args:
        y_true: True target values.
        y_pred: Predicted mean values.
    """
    import matplotlib.pyplot as plt

    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.6, color="purple")
    plt.axhline(0, color="red", linestyle="--", linewidth=2)
    plt.title("Residual Plot")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.grid()
    plt.show()


def plot_all_features(X_train, y_train, X_test, mean, std):
    """
    Plots confidence intervals for all features.

    Args:
        X_train: Training features.
        y_train: Training targets.
        X_test: Test features.
        mean: Predicted mean values.
        std: Predicted standard deviations.
    """
    n_features = X_train.shape[1]
    for feature_index in range(n_features):
        plot_confidence_intervals_per_feature(X_train, y_train, X_test, mean, std, feature_index)


def plot_predicted_vs_true(y_true, y_pred):
    """
    Plots predicted values against true values.

    Args:
        y_true: True target values.
        y_pred: Predicted mean values.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, color="green")
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color="red", linestyle="--")
    plt.title("Predicted vs True Values")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.grid()
    plt.show()


def plot_residual_distribution(y_true, y_pred):
    """
    Plots the distribution of residuals.

    Args:
        y_true: True target values.
        y_pred: Predicted mean values.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, color="orange", bins=30)
    plt


def plot_kernel_matrix(kernel, X):
    """
    Plots the kernel matrix as a heatmap.

    Args:
        kernel: Kernel used in the GP model.
        X: Input data to compute the kernel matrix.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    K = kernel(torch.tensor(X, dtype=torch.float32), torch.tensor(X, dtype=torch.float32)).evaluate()
    plt.figure(figsize=(10, 6))
    plt.imshow(K.detach().numpy(), interpolation="nearest", cmap="viridis")
    plt.title("Kernel Matrix Heatmap")
    plt.colorbar(label="Covariance")
    plt.xlabel("Data Points")
    plt.ylabel("Data Points")
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)

    # Generate synthetic data
    n_samples, n_features = 500, 5
    X = np.random.rand(n_samples, n_features) * 10  # Random features in range [0, 10]
    true_beta = np.array([3.5, -2.1, 0.8, 4.2, -1.5])  # True coefficients
    noise = np.random.normal(0, 2.0, size=n_samples)  # Gaussian noise

    # Linear relationship
    y = X @ true_beta + noise

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the GP model
    gpr = GaussianProcessRegression()
    gpr.train(X_train, y_train, num_epochs=50, lr=0.1)

    # Predict
    mean, std = gpr.predict(X_test)

    # Visualize
    plot_loss_curve(gpr.loss_values)  # Loss curve
    plot_residuals(y_test, mean)  # Residuals
    plot_predicted_vs_true(y_test, mean)  # Predicted vs True
    plot_residual_distribution(y_test, mean)  # Residual distribution
    plot_all_features(X_train, y_train, X_test, mean, std)  # Confidence intervals for all features
    plot_kernel_matrix(gpr.model.covar_module.base_kernel, X_test)  # Kernel matrix heatmap

