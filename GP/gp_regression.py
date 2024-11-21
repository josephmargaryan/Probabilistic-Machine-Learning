import torch
import gpytorch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel=None):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel or gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class GaussianProcessRegression:
    def __init__(self, kernel=None, noise=0.1):
        self.kernel = kernel if kernel else gpytorch.kernels.RBFKernel()
        self.noise = noise
        self.loss_values = []

    def train(self, X_train, y_train, num_epochs=50, lr=0.1):
        train_x = torch.tensor(X_train, dtype=torch.float32)
        train_y = torch.tensor(y_train, dtype=torch.float32)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = GPRegressionModel(train_x, train_y, self.likelihood, kernel=self.kernel)
        self.likelihood.noise_covar.initialize(noise=self.noise)

        # Train mode
        self.model.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            output = self.model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            self.loss_values.append(loss.item())
            print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {loss.item()}")
            optimizer.step()

    def predict(self, X_test):
        self.model.eval()
        self.likelihood.eval()
        test_x = torch.tensor(X_test, dtype=torch.float32)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            output = self.likelihood(self.model(test_x))
            mean = output.mean.numpy()
            std = output.variance.sqrt().numpy()
            return mean, std


if __name__ == "__main__":
    # Generate synthetic regression data
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split

    X, y = make_regression(n_samples=500, n_features=1, noise=10.0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Gaussian Process Regression
    gpr = GaussianProcessRegression()
    gpr.train(X_train, y_train, num_epochs=50, lr=0.1)

    # Predict and evaluate
    mean, std = gpr.predict(X_test)
    mse = mean_squared_error(y_test, mean)
    mae = mean_absolute_error(y_test, mean)
    r2 = r2_score(y_test, mean)

    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R^2: {r2:.4f}")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, y_train, color="blue", label="Training Data")
    plt.plot(X_test, mean, color="red", label="Mean Prediction")
    plt.fill_between(
        X_test.flatten(),
        mean - 1.96 * std,
        mean + 1.96 * std,
        color="pink",
        alpha=0.3,
        label="95% Confidence Interval",
    )
    plt.title("Gaussian Process Regression")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.show()
