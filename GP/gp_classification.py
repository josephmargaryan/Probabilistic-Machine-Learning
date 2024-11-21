import torch
import gpytorch
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

class GPClassificationModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, kernel=None):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(GPClassificationModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel or gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class GaussianProcessClassification:
    def __init__(self, kernel=None, num_inducing_points=100):
        self.kernel = kernel if kernel else gpytorch.kernels.RBFKernel()
        self.num_inducing_points = num_inducing_points
        self.loss_values = []

    def train(self, X_train, y_train, num_epochs=50, lr=0.1):
        train_x = torch.tensor(X_train, dtype=torch.float32)
        train_y = torch.tensor(y_train, dtype=torch.float32)
        inducing_points = train_x[:min(self.num_inducing_points, train_x.size(0))]
        self.likelihood = gpytorch.likelihoods.BernoulliLikelihood()
        self.model = GPClassificationModel(inducing_points, kernel=self.kernel)

        # Train mode
        self.model.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=train_y.size(0))

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
            probs = torch.sigmoid(output.mean).numpy()
            return probs

if __name__ == "__main__":

    # Generate synthetic classification data
    X, y = make_classification(n_samples=500, n_features=5, n_informative=3, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Gaussian Process Classification
    gpc = GaussianProcessClassification()
    gpc.train(X_train, y_train, num_epochs=50, lr=0.1)

    # Predict and evaluate
    probs = gpc.predict(X_test)
    y_pred = (probs > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Classification Accuracy: {accuracy:.4f}")
