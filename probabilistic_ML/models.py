import numpy as np
import random
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.fft import fft, ifft
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import gpytorch
from itertools import product
import torch.distributions as dist
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
import networkx as nx
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pyro
import pyro.distributions as pyro_dist
from pyro.infer import MCMC, NUTS


class BayesianNetwork:
    def __init__(self):
        # Initialize the Bayesian Network structure and CPTs
        self.structure = {}
        self.cpt = {}
        self.parameters = {}

    def add_node(self, node):
        # Add a node to the network
        if node not in self.structure:
            self.structure[node] = []
        else:
            print(f"Node {node} already exists.")

    def add_edge(self, parent, child):
        # Add a directed edge from parent to child
        if child not in self.structure:
            self.structure[child] = []
        self.structure[child].append(parent)

    def set_cpt(self, node, cpt):
        # Set the conditional probability table (CPT) for a node
        self.cpt[node] = cpt

    def get_parents(self, node):
        # Get parents of a node
        return self.structure.get(node, [])

    def calculate_joint_probability(self, evidence):
        # Calculate the joint probability for given evidence
        joint_prob = 1.0
        for node in self.structure:
            if node in evidence:
                parents = self.get_parents(node)
                if parents:
                    parent_values = tuple(evidence[parent] for parent in parents)
                    prob = self.cpt[node][parent_values][evidence[node]]
                else:
                    prob = self.cpt[node][()][evidence[node]]
                joint_prob *= prob
        return joint_prob

    def fit_mle(self, data):
        # Maximum Likelihood Estimation
        counts = {}
        for node in self.structure:
            parents = self.get_parents(node)
            counts[node] = {}
            # Generate all possible combinations of parent states
            all_parent_combinations = list(product([True, False], repeat=len(parents)))

            for parent_values in all_parent_combinations:
                counts[node][parent_values] = {True: 0, False: 0}

            for observation in data:
                parent_values = tuple(observation[parent] for parent in parents)
                counts[node][parent_values][observation[node]] += 1

        for node in counts:
            self.cpt[node] = {}
            for parent_values in counts[node]:
                total = sum(counts[node][parent_values].values())
                if total > 0:
                    self.cpt[node][parent_values] = {
                        outcome: count / total
                        for outcome, count in counts[node][parent_values].items()
                    }
                else:
                    # Assign a uniform distribution if no data is available
                    self.cpt[node][parent_values] = {True: 0.5, False: 0.5}

    def fit_em(self, data, max_iter=100):
        # Expectation-Maximization Algorithm
        for _ in range(max_iter):
            expected_counts = self.e_step(data)
            self.m_step(expected_counts)

    def e_step(self, data):
        # Estimate expected counts
        expected_counts = {}
        for node in self.structure:
            parents = self.get_parents(node)
            expected_counts[node] = {}
            for observation in data:
                parent_values = tuple(observation[parent] for parent in parents)
                if parent_values not in expected_counts[node]:
                    expected_counts[node][parent_values] = {True: 0, False: 0}
                expected_counts[node][parent_values][
                    observation[node]
                ] += 1  # Simplified for full observation data
        return expected_counts

    def m_step(self, expected_counts):
        # Update parameters based on expected counts
        for node in expected_counts:
            self.cpt[node] = {}
            for parent_values in expected_counts[node]:
                total = sum(expected_counts[node][parent_values].values())
                self.cpt[node][parent_values] = {
                    outcome: count / total
                    for outcome, count in expected_counts[node][parent_values].items()
                }

    def fit_mcmc(self, data, num_samples=1000):
        # Markov Chain Monte Carlo (Gibbs Sampling)
        samples = []
        current_state = {node: random.choice([True, False]) for node in self.structure}
        for _ in range(num_samples):
            for node in current_state:
                parents = self.get_parents(node)
                parent_values = tuple(current_state[parent] for parent in parents)
                # Check if the parent_values exist in the CPT
                if parent_values not in self.cpt[node]:
                    # Handle missing keys with a safe default
                    prob_true = 0.5  # Default probability for missing entries
                    prob_false = 0.5
                else:
                    prob_true = self.cpt[node][parent_values].get(True, 0.5)
                    prob_false = self.cpt[node][parent_values].get(False, 0.5)

                # Normalize the probabilities to avoid issues
                total_prob = prob_true + prob_false
                if total_prob > 0:
                    prob_true /= total_prob
                    prob_false /= total_prob

                current_state[node] = np.random.choice(
                    [True, False], p=[prob_true, prob_false]
                )
            samples.append(current_state.copy())
        return samples

    def fit_variational_inference(self, data, max_iter=100):
        # Variational Inference
        def kl_divergence(q, p):
            return np.sum(q * np.log(q / p))

        for node in self.structure:
            parents = self.get_parents(node)
            self.parameters[node] = {
                parent_values: {True: 0.5, False: 0.5}
                for parent_values in self.cpt[node]
            }

        for _ in range(max_iter):
            for node in self.parameters:
                parents = self.get_parents(node)
                for parent_values in self.parameters[node]:
                    self.parameters[node][parent_values][True] = np.random.rand()
                    self.parameters[node][parent_values][False] = (
                        1 - self.parameters[node][parent_values][True]
                    )

        return self.parameters

    def visualize_network(self):
        # Visualize the Bayesian Network structure
        graph = nx.DiGraph()
        for child, parents in self.structure.items():
            for parent in parents:
                graph.add_edge(parent, child)

        plt.figure(figsize=(8, 6))
        nx.draw_networkx(
            graph,
            with_labels=True,
            node_size=2000,
            node_color="lightblue",
            font_size=10,
            font_weight="bold",
        )
        plt.title("Bayesian Network Structure")
        plt.show()

    def combine_evidence(self, initial_evidence, new_evidence):
        # Combine and update evidence
        combined_evidence = initial_evidence.copy()
        combined_evidence.update(new_evidence)
        return combined_evidence

    def predict(self, target_node, evidence):
        # Predict the probability distribution for a given target node, given partial evidence
        if target_node not in self.structure:
            raise ValueError("Target node not found in the Bayesian Network")

        # Get parents of the target node
        parents = self.get_parents(target_node)
        parent_values = tuple(evidence.get(parent, None) for parent in parents)

        if parent_values not in self.cpt[target_node]:
            raise ValueError(
                "Parent combination not found in CPT. Please provide full evidence or train on more data."
            )

        # Calculate probabilities for True and False outcomes
        prob_true = self.cpt[target_node][parent_values].get(True, 0.5)
        prob_false = self.cpt[target_node][parent_values].get(False, 0.5)

        # Normalize to ensure they sum to 1
        total_prob = prob_true + prob_false
        if total_prob > 0:
            prob_true /= total_prob
            prob_false /= total_prob

        return {True: prob_true, False: prob_false}


# The class now includes methods for graph visualization, evidence combination, and is ready to be extended for continuous data handling.


# Implementing Gaussian Process with FFT
class FFTGaussianProcess:
    def __init__(self, kernel_size, kernel_type="RBF", noise_level=1e-3):
        self.kernel_size = kernel_size
        self.kernel_type = kernel_type
        self.noise_level = noise_level
        self.kernel_matrix = self._generate_kernel()

    def _generate_kernel(self):
        if self.kernel_type == "RBF":
            kernel = np.exp(-np.linspace(0, 1, self.kernel_size) ** 2)
        elif self.kernel_type == "Matern":
            kernel = (1 + np.sqrt(3) * np.linspace(0, 1, self.kernel_size)) * np.exp(
                -np.sqrt(3) * np.linspace(0, 1, self.kernel_size)
            )
        else:
            raise ValueError("Unsupported kernel type")
        return np.fft.fft(kernel)

    def predict(self, x):
        fft_x = np.fft.fft(x)
        result = np.fft.ifft(fft_x * self.kernel_matrix).real
        result += self.noise_level * np.random.randn(*result.shape)  # Adding noise
        return result


# Implementing Convolutional Model using FFT
class FFTConvolutionalLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(FFTConvolutionalLayer, self).__init__()
        self.kernel = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size)
        )  # Learnable kernel
        self.padding = nn.ConstantPad1d((kernel_size // 2, kernel_size // 2), 0)

    def forward(self, x):
        x_padded = self.padding(x)
        x_fft = torch.fft.fft(x_padded, dim=-1)
        kernel_fft = torch.fft.fft(self.kernel, dim=-1)

        kernel_fft = torch.fft.fft(
            torch.nn.functional.pad(
                self.kernel, (0, x_fft.shape[-1] - self.kernel.shape[-1])
            ),
            dim=-1,
        )

        convolved = torch.fft.ifft(x_fft * kernel_fft, dim=-1).real
        return torch.relu(convolved)  # Adding activation


# Bayesian Neural Network leveraging circulant matrices
class BayesianNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BayesianNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.circulant_matrix = torch.randn(input_size, input_size)

    def forward(self, x):
        fft_x = torch.fft.fft(x)
        circulant_fft = torch.fft.fft(self.circulant_matrix)

        fft_x = fft_x.unsqueeze(0)
        out = torch.fft.ifft(torch.matmul(fft_x, circulant_fft)).real
        out = out.squeeze(0)

        out = self.fc1(out)
        out = torch.relu(out)
        out = self.fc2(out)

        # Sample output for Bayesian inference
        mean = out
        std = torch.exp(0.5 * out)  # Assuming the output layer's standard deviation
        sampled_output = mean + std * torch.randn_like(std)
        return sampled_output


class StandardGaussianProcess:
    def __init__(self, kernel_type="RBF", noise_level=1e-3):
        if kernel_type == "RBF":
            self.kernel = RBF()
        elif kernel_type == "Matern":
            self.kernel = Matern()
        else:
            raise ValueError("Unsupported kernel type")
        self.model = GaussianProcessRegressor(kernel=self.kernel, alpha=noise_level)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X, return_std=True)


class StandardConvolutionalLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(StandardConvolutionalLayer, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        return self.activation(x)

    def train_model(self, data_loader, criterion, optimizer, epochs=10):
        self.train()
        for epoch in range(epochs):
            for inputs, targets in data_loader:
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")


class BayesianLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(BayesianLayer, self).__init__()
        self.mu = nn.Parameter(torch.randn(in_features, out_features) * 0.1)
        self.rho = nn.Parameter(torch.randn(in_features, out_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        std = torch.log1p(
            torch.exp(self.rho)
        )  # Softplus to ensure positive std deviation
        weights_distribution = dist.Normal(self.mu, std)
        weights = weights_distribution.rsample()
        return torch.matmul(x, weights) + self.bias


class StandardBayesianNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(StandardBayesianNeuralNetwork, self).__init__()
        self.bayesian_layer1 = BayesianLayer(input_size, hidden_size)
        self.activation = nn.ReLU()
        self.bayesian_layer2 = BayesianLayer(hidden_size, output_size)

    def forward(self, x):
        x = self.bayesian_layer1(x)
        x = self.activation(x)
        x = self.bayesian_layer2(x)
        return x

    def train_model(self, data_loader, criterion, optimizer, epochs=10):
        self.train()
        for epoch in range(epochs):
            for inputs, targets in data_loader:
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

class BayesianNeuralNetworkMCMC(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BayesianNeuralNetworkMCMC, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.circulant_matrix = torch.randn(input_size, input_size)

    def model(self, x, y=None):
        # Define priors for weights and biases using Normal distributions
        fc1_weight_prior = pyro_dist.Normal(torch.zeros_like(self.fc1.weight), torch.ones_like(self.fc1.weight))
        fc1_bias_prior = pyro_dist.Normal(torch.zeros_like(self.fc1.bias), torch.ones_like(self.fc1.bias))
        fc2_weight_prior = pyro_dist.Normal(torch.zeros_like(self.fc2.weight), torch.ones_like(self.fc2.weight))
        fc2_bias_prior = pyro_dist.Normal(torch.zeros_like(self.fc2.bias), torch.ones_like(self.fc2.bias))

        # Sample weights and biases
        fc1_weight = pyro.sample("fc1_weight", fc1_weight_prior)
        fc1_bias = pyro.sample("fc1_bias", fc1_bias_prior)
        fc2_weight = pyro.sample("fc2_weight", fc2_weight_prior)
        fc2_bias = pyro.sample("fc2_bias", fc2_bias_prior)

        # Forward pass using sampled weights and biases
        hidden = torch.matmul(x, fc1_weight.T) + fc1_bias
        hidden = torch.relu(hidden)
        output = torch.matmul(hidden, fc2_weight.T) + fc2_bias

        # Define likelihood
        with pyro.plate("data", size=x.shape[0]):
            pyro.sample("obs", pyro_dist.Normal(output, 0.1), obs=y)

    def guide(self, x, y=None):
        # Variational parameters for each weight and bias
        fc1_weight_mu = pyro.param("fc1_weight_mu", torch.randn_like(self.fc1.weight))
        fc1_weight_sigma = pyro.param("fc1_weight_sigma", torch.ones_like(self.fc1.weight), constraint=pyro.distributions.constraints.positive)
        fc1_bias_mu = pyro.param("fc1_bias_mu", torch.randn_like(self.fc1.bias))
        fc1_bias_sigma = pyro.param("fc1_bias_sigma", torch.ones_like(self.fc1.bias), constraint=pyro.distributions.constraints.positive)
        fc2_weight_mu = pyro.param("fc2_weight_mu", torch.randn_like(self.fc2.weight))
        fc2_weight_sigma = pyro.param("fc2_weight_sigma", torch.ones_like(self.fc2.weight), constraint=pyro.distributions.constraints.positive)
        fc2_bias_mu = pyro.param("fc2_bias_mu", torch.randn_like(self.fc2.bias))
        fc2_bias_sigma = pyro.param("fc2_bias_sigma", torch.ones_like(self.fc2.bias), constraint=pyro.distributions.constraints.positive)

        # Sample weights and biases from variational distributions
        pyro.sample("fc1_weight", pyro_dist.Normal(fc1_weight_mu, fc1_weight_sigma))
        pyro.sample("fc1_bias", pyro_dist.Normal(fc1_bias_mu, fc1_bias_sigma))
        pyro.sample("fc2_weight", pyro_dist.Normal(fc2_weight_mu, fc2_weight_sigma))
        pyro.sample("fc2_bias", pyro_dist.Normal(fc2_bias_mu, fc2_bias_sigma))

    def forward_fft(self, x):
        # Use FFT for circulant matrix multiplication
        fft_x = fft.fft(x)
        circulant_fft = fft.fft(self.circulant_matrix)
        out = fft.ifft(fft_x * circulant_fft).real
        return out
    
# Running MCMC with NUTS

def run_mcmc(x_train, y_train):
    bnn = BayesianNeuralNetworkMCMC(input_size=x_train.shape[1], hidden_size=20, output_size=1)
    nuts_kernel = NUTS(bnn.model)
    mcmc = MCMC(nuts_kernel, num_samples=500, warmup_steps=100)
    mcmc.run(x_train, y_train)
    samples = mcmc.get_samples()
    return samples

# Uncertainty visualization function
def plot_uncertainty(x_train, y_train, x_test, samples):
    predictions = []
    for i in range(100):  # Sample 100 different parameter sets
        fc1_weight = samples['fc1_weight'][i]
        fc1_bias = samples['fc1_bias'][i]
        fc2_weight = samples['fc2_weight'][i]
        fc2_bias = samples['fc2_bias'][i]

        hidden = torch.matmul(x_test, fc1_weight.T) + fc1_bias
        hidden = torch.relu(hidden)
        output = torch.matmul(hidden, fc2_weight.T) + fc2_bias
        predictions.append(output.detach().numpy())

    predictions = np.array(predictions)
    mean_prediction = predictions.mean(axis=0)
    std_prediction = predictions.std(axis=0)

    plt.figure(figsize=(10, 6))
    plt.fill_between(np.arange(len(mean_prediction)), mean_prediction - std_prediction, mean_prediction + std_prediction, color='lightblue', alpha=0.5, label='Uncertainty')
    plt.plot(mean_prediction, label='Mean Prediction', color='blue')
    plt.scatter(np.arange(len(y_train)), y_train, color='red', label='True Data')
    plt.title("Posterior Predictive Distribution with Uncertainty")
    plt.xlabel("Data Points")
    plt.ylabel("Predicted Value")
    plt.legend()
    plt.show()

def test_bayesian_neural_network_mcmc():
    # Generate synthetic data
    x_train = torch.randn(50, 10)
    y_train = torch.sin(x_train.sum(dim=1)) + 0.1 * torch.randn(50)  # Noisy sine data

    # Run MCMC
    print("\nRunning MCMC with NUTS...")
    samples = run_mcmc(x_train, y_train)
    print("MCMC samples:", samples)

    # Analyze sample statistics (mean of weights as an example)
    for param_name, param_samples in samples.items():
        print(f"Mean of {param_name}: {param_samples.mean().item()}")

    # Generate test data for predictions
    x_test = torch.randn(25, 10)
    plot_uncertainty(x_train, y_train, x_test, samples)


def test_standard_models():
    # Test Standard Gaussian Process
    print("\nTesting Standard Gaussian Process...")
    gp_model = StandardGaussianProcess(kernel_type="RBF")
    X = np.linspace(-5, 5, 100).reshape(-1, 1)
    y = np.sin(X).ravel() + 0.1 * np.random.randn(100)
    gp_model.fit(X, y)
    X_test = np.linspace(-6, 6, 200).reshape(-1, 1)
    gp_predictions, gp_std = gp_model.predict(X_test)
    plt.fill_between(
        X_test.ravel(), gp_predictions - gp_std, gp_predictions + gp_std, alpha=0.3
    )
    plt.plot(X_test, gp_predictions, label="GP Prediction")
    plt.scatter(X, y, c="r", marker="x", label="Training Data")
    plt.title("Standard Gaussian Process Predictions")
    plt.legend()
    plt.show()

    # Test Standard Convolutional Layer
    print("\nTesting Standard Convolutional Layer...")
    conv_model = StandardConvolutionalLayer(
        in_channels=1, out_channels=1, kernel_size=3
    )
    data = torch.sin(torch.linspace(0, 4 * np.pi, 100)).reshape(1, 1, -1)
    target = torch.cos(torch.linspace(0, 4 * np.pi, 100)).reshape(1, 1, -1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(conv_model.parameters(), lr=0.01)
    dataset = TensorDataset(data, target)
    data_loader = DataLoader(dataset, batch_size=10)
    conv_model.train_model(data_loader, criterion, optimizer, epochs=5)

    # Test Standard Bayesian Neural Network
    print("\nTesting Standard Bayesian Neural Network...")
    bnn_model = StandardBayesianNeuralNetwork(
        input_size=10, hidden_size=20, output_size=1
    )
    x_input = torch.randn(10)
    y_target = torch.tensor([1.0])  # Dummy target for training
    bnn_criterion = nn.MSELoss()
    bnn_optimizer = optim.Adam(bnn_model.parameters(), lr=0.01)
    bnn_dataset = TensorDataset(x_input.unsqueeze(0), y_target.unsqueeze(0))
    bnn_data_loader = DataLoader(bnn_dataset, batch_size=1)
    bnn_model.train_model(bnn_data_loader, bnn_criterion, bnn_optimizer, epochs=5)
    bnn_result = bnn_model(x_input)
    print("BNN Output after training:", bnn_result)


def test_models():
    # Test FFT Gaussian Process
    print("\nTesting FFT Gaussian Process...")
    gp_model = FFTGaussianProcess(kernel_size=100)
    x = np.random.randn(100)
    gp_result = gp_model.predict(x)
    print("GP Prediction:", gp_result)

    # Test Convolutional Model using FFT
    print("\nTesting FFT Convolutional Layer...")
    conv_model = FFTConvolutionalLayer(in_channels=1, out_channels=1, kernel_size=5)
    x_tensor = torch.randn(1, 1, 10)
    conv_result = conv_model(x_tensor)
    print("Convolution Result:", conv_result)

    # Visualize the convolution result
    plt.plot(conv_result.detach().numpy().flatten(), label="Convolution Result")
    plt.title("FFT Convolutional Layer Output")
    plt.legend()
    plt.show()

    # Test Bayesian Neural Network
    print("\nTesting Bayesian Neural Network...")
    bnn_model = BayesianNeuralNetwork(input_size=10, hidden_size=20, output_size=1)
    x_input = torch.randn(10)
    bnn_result = bnn_model(x_input)
    print("BNN Output:", bnn_result)

    # Test Bayesian Network visualization
    print("\nTesting Bayesian Network Visualization...")
    bn = BayesianNetwork()
    bn.add_node("Rain")
    bn.add_node("Sprinkler")
    bn.add_node("WetGrass")
    bn.add_edge("Rain", "WetGrass")
    bn.add_edge("Sprinkler", "WetGrass")
    bn.visualize_network()

    # Test Evidence Combination
    print("\nTesting Evidence Combination...")
    initial_evidence = {"Rain": True}
    new_evidence = {"Sprinkler": False}
    combined_evidence = bn.combine_evidence(initial_evidence, new_evidence)
    print("Combined evidence:", combined_evidence)

    # Test Prediction with Combined Evidence
    print("\nTesting Prediction with Combined Evidence...")
    # Set dummy CPTs for testing
    bn.set_cpt("Rain", {(): {True: 0.2, False: 0.8}})
    bn.set_cpt("Sprinkler", {(): {True: 0.5, False: 0.5}})
    bn.set_cpt(
        "WetGrass",
        {
            (True, True): {True: 0.99, False: 0.01},
            (True, False): {True: 0.9, False: 0.1},
            (False, True): {True: 0.8, False: 0.2},
            (False, False): {True: 0.0, False: 1.0},
        },
    )
    prediction_combined = bn.predict("WetGrass", combined_evidence)
    print(
        f"Prediction for 'WetGrass' with combined evidence {combined_evidence}: {prediction_combined}"
    )


def test_bayesian_network():
    print("\nTesting Bayesian Network Optimization Methods...")
    # Create a simple Bayesian Network
    bn = BayesianNetwork()
    bn.add_node("Rain")
    bn.add_node("Sprinkler")
    bn.add_node("WetGrass")

    bn.add_edge("Rain", "WetGrass")
    bn.add_edge("Sprinkler", "WetGrass")

    # Set dummy CPTs for testing
    bn.set_cpt("Rain", {(): {True: 0.2, False: 0.8}})
    bn.set_cpt("Sprinkler", {(): {True: 0.5, False: 0.5}})
    bn.set_cpt(
        "WetGrass",
        {
            (True, True): {True: 0.99, False: 0.01},
            (True, False): {True: 0.9, False: 0.1},
            (False, True): {True: 0.8, False: 0.2},
            (False, False): {True: 0.0, False: 1.0},
        },
    )

    # Create sample data
    sample_data = [
        {"Rain": True, "Sprinkler": False, "WetGrass": True},
        {"Rain": False, "Sprinkler": True, "WetGrass": True},
        {"Rain": True, "Sprinkler": True, "WetGrass": False},
    ]

    # Test MLE
    print("\nTesting MLE...")
    bn.fit_mle(sample_data)
    print("Updated CPTs after MLE:", bn.cpt)

    # Test EM
    print("\nTesting EM...")
    bn.fit_em(sample_data, max_iter=10)
    print("Updated CPTs after EM:", bn.cpt)

    # Test MCMC
    print("\nTesting MCMC...")
    samples = bn.fit_mcmc(sample_data, num_samples=100)
    print("Sampled states from MCMC:", samples[:5])  # Print first 5 samples

    # Test Variational Inference
    print("\nTesting Variational Inference...")
    params = bn.fit_variational_inference(sample_data, max_iter=10)
    print("Updated parameters after Variational Inference:", params)

    # Test Predictions
    print("\nTesting Predictions...")
    evidence = {"Rain": True, "Sprinkler": False}
    prediction = bn.predict("WetGrass", evidence)
    print(f"Prediction for 'WetGrass' given evidence {evidence}: {prediction}")

    # Test Graph Visualization
    print("\nTesting Graph Visualization...")
    bn.visualize_network()

    # Test Evidence Combination
    print("\nTesting Evidence Combination...")
    initial_evidence = {"Rain": True}
    new_evidence = {"Sprinkler": False}
    combined_evidence = bn.combine_evidence(initial_evidence, new_evidence)
    print(f"Combined evidence: {combined_evidence}")

    # Test Prediction with Combined Evidence
    prediction_combined = bn.predict("WetGrass", combined_evidence)
    print(
        f"Prediction for 'WetGrass' with combined evidence {combined_evidence}: {prediction_combined}"
    )


def test_both_models():
    # Test Standard and FFT Gaussian Process
    print("\nTesting Standard and FFT Gaussian Process...")
    X = np.linspace(-5, 5, 100).reshape(-1, 1)
    y = np.sin(X).ravel() + 0.1 * np.random.randn(100)

    # Standard Gaussian Process
    gp_model_standard = StandardGaussianProcess(kernel_type="RBF")
    gp_model_standard.fit(X, y)
    X_test = np.linspace(-6, 6, 200).reshape(-1, 1)
    gp_predictions_standard, gp_std_standard = gp_model_standard.predict(X_test)

    # FFT Gaussian Process
    gp_model_fft = FFTGaussianProcess(kernel_size=100)
    gp_predictions_fft = gp_model_fft.predict(np.sin(np.linspace(0, 2 * np.pi, 100)))

    # Plot the results
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.fill_between(
        X_test.ravel(),
        gp_predictions_standard - gp_std_standard,
        gp_predictions_standard + gp_std_standard,
        alpha=0.3,
    )
    plt.plot(X_test, gp_predictions_standard, label="Standard GP Prediction")
    plt.scatter(X, y, c="r", marker="x", label="Training Data")
    plt.title("Standard Gaussian Process")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(gp_predictions_fft, label="FFT GP Prediction")
    plt.title("FFT Gaussian Process")
    plt.legend()
    plt.show()

    # Test Standard and FFT Convolutional Layer
    print("\nTesting Standard and FFT Convolutional Layer...")
    conv_model_standard = StandardConvolutionalLayer(
        in_channels=1, out_channels=1, kernel_size=3
    )
    conv_model_fft = FFTConvolutionalLayer(in_channels=1, out_channels=1, kernel_size=3)
    x_tensor = torch.randn(1, 1, 100)

    # Forward pass
    conv_result_standard = conv_model_standard(x_tensor)
    conv_result_fft = conv_model_fft(x_tensor)

    # Plot the convolution results
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(
        conv_result_standard.detach().numpy().flatten(), label="Standard Conv Result"
    )
    plt.title("Standard Convolutional Layer Output")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(conv_result_fft.detach().numpy().flatten(), label="FFT Conv Result")
    plt.title("FFT Convolutional Layer Output")
    plt.legend()
    plt.show()

    # Test Standard and FFT Bayesian Neural Network
    print("\nTesting Standard and FFT Bayesian Neural Network...")
    bnn_model_standard = StandardBayesianNeuralNetwork(
        input_size=10, hidden_size=20, output_size=1
    )
    bnn_model_fft = BayesianNeuralNetwork(input_size=10, hidden_size=20, output_size=1)
    x_input = torch.randn(10)

    # Forward pass
    bnn_result_standard = bnn_model_standard(x_input)
    bnn_result_fft = bnn_model_fft(x_input)

    print("Standard BNN Output:", bnn_result_standard)
    print("FFT BNN Output:", bnn_result_fft)



"""probabilistic_ml/
├── __init__.py
├── bayesian_network.py
├── fft_gaussian_process.py
├── fft_convolutional_layer.py
├── bayesian_neural_network.py
├── standard_gaussian_process.py
├── standard_convolutional_layer.py
├── bayesian_layer.py
├── standard_bayesian_neural_network.py
└── bayesian_neural_network_mcmc.py
"""

if __name__ == "__main__":
    test_standard_models()
    test_models()
    test_bayesian_network()
    test_both_models()
    test_bayesian_neural_network_mcmc()
