from gp_classification import GaussianProcessClassification
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

def plot_loss_curve(loss_values):
    """
    Plots the loss curve for GPC training.

    Args:
        loss_values: List of loss values recorded during training.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(loss_values)), loss_values, label="ELBO", color="blue")
    plt.title("Evidence Lower Bound (ELBO) During Training")
    plt.xlabel("Epochs")
    plt.ylabel("ELBO")
    plt.legend()
    plt.grid()
    plt.show()


def plot_prediction_confidence(probs, y_true):
    """
    Plots the distribution of predicted probabilities for each class.

    Args:
        probs: Predicted probabilities for the positive class.
        y_true: True labels.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 6))
    sns.histplot(probs[y_true == 1], bins=30, color="green", label="Class 1", kde=True, alpha=0.6)
    sns.histplot(probs[y_true == 0], bins=30, color="red", label="Class 0", kde=True, alpha=0.6)
    plt.title("Prediction Confidence Distribution")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid()
    plt.show()


def plot_roc_curve(y_true, probs):
    """
    Plots the ROC curve.

    Args:
        y_true: True labels.
        probs: Predicted probabilities for the positive class.
    """
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=2, label="Random Classifier")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()


def plot_confusion_matrix(y_true, y_pred):
    """
    Plots the confusion matrix.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
    """
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])

    disp.plot(cmap="Blues")
    disp.ax_.set_title("Confusion Matrix")
    disp.ax_.grid(False)
    plt.show()


def plot_feature_contributions(X, probs, feature_index=0):
    """
    Plots predicted probabilities against a single feature.

    Args:
        X: Input features.
        probs: Predicted probabilities for the positive class.
        feature_index: Index of the feature to plot (default is 0).
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, feature_index], probs, color="blue", alpha=0.6)
    plt.title(f"Feature Contribution to Prediction (Feature {feature_index})")
    plt.xlabel(f"Feature {feature_index}")
    plt.ylabel("Predicted Probability")
    plt.grid()
    plt.show()


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

def plot_binary_calibration_curve(y_true, probs, n_bins=10):
    """
    Plots the calibration curve for binary classification.

    Args:
        y_true: True binary labels (0 or 1).
        probs: Predicted probabilities for the positive class.
        n_bins: Number of bins for calibration curve.
    """
    # Compute calibration curve
    true_probs, mean_pred_probs = calibration_curve(y_true, probs, n_bins=n_bins, strategy="uniform")

    # Plot calibration curve
    plt.figure(figsize=(10, 6))
    plt.plot(mean_pred_probs, true_probs, marker="o", label="Calibration Curve")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    plt.title("Calibration Curve for Binary Classification")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.legend()
    plt.grid()
    plt.show()





if __name__ == "__main__":
    # Generate synthetic classification data
    X, y = make_classification(n_samples=500, n_features=5, n_informative=3, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Gaussian Process Classification
    gpc = GaussianProcessClassification()
    gpc.train(X_train, y_train, num_epochs=50, lr=0.1)


    # After training and prediction
    probs = gpc.predict(X_test)
    y_pred = (probs > 0.5).astype(int)

    # Visualizations
    plot_loss_curve(gpc.loss_values)  # Loss curve
    plot_prediction_confidence(probs, y_test)  # Confidence distribution
    plot_roc_curve(y_test, probs)  # ROC curve
    plot_confusion_matrix(y_test, y_pred)  # Confusion matrix
    plot_feature_contributions(X_test, probs, feature_index=0)  # Feature contributions
    plot_kernel_matrix(gpc.model.covar_module.base_kernel, X_test)  # Kernel matrix heatmap
    plot_binary_calibration_curve(y_test, probs)
