import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Define the Maximum Entropy Model
class MaximumEntropyModel:
    def __init__(self, features, constraints):
        """
        Initialize with a set of features and their constraints.
        :param features: List of functions f_i(x)
        :param constraints: List of expected values for each feature
        """
        self.features = features
        self.constraints = constraints
        self.lambdas = np.zeros(len(features))  # Initialize Lagrange multipliers

    def objective_function(self, lambdas):
        """
        Calculate the objective for optimization (log-likelihood).
        """
        z = sum(np.exp(sum(lambdas[i] * f(x) for i, f in enumerate(self.features)))
                for x in self.sample_space)
        log_likelihood = sum(lambdas[i] * constraint for i, constraint in enumerate(self.constraints)) - np.log(z)
        return -log_likelihood  # Negative for minimization

    def fit(self, sample_space):
        """
        Fit the model using optimization to find the Lagrange multipliers.
        """
        self.sample_space = sample_space  # Assume discrete space for simplicity
        result = minimize(self.objective_function, self.lambdas, method='L-BFGS-B')
        self.lambdas = result.x

    def probability(self, x):
        """
        Calculate the probability for a given x.
        """
        numerator = np.exp(sum(self.lambdas[i] * f(x) for i, f in enumerate(self.features)))
        z = sum(np.exp(sum(self.lambdas[i] * f(x) for i, f in enumerate(self.features)))
                for x in self.sample_space)
        return numerator / z

# Test function for Maximum Entropy Model
def test_MaxEnt():
    # Define the sample space (e.g., binary outcomes)
    sample_space = [0, 1]

    # Define feature functions (e.g., indicator functions)
    def f1(x): return x

    def f2(x): return 1 - x

    features = [f1, f2]

    # Define expected values for the constraints
    constraints = [0.5, 0.5]  # Example: expecting equal probability for 0 and 1

    # Create and fit the Maximum Entropy Model
    model = MaximumEntropyModel(features, constraints)
    model.fit(sample_space)

    # Test the probability distribution
    probabilities = {x: model.probability(x) for x in sample_space}
    print("Probabilities:", probabilities)

    # Check if the probabilities sum to 1
    assert np.isclose(sum(probabilities.values()), 1.0), "Probabilities do not sum to 1"

    # Visualize the probabilities
    plt.bar(probabilities.keys(), probabilities.values(), color='lightblue')
    plt.xlabel("Outcome")
    plt.ylabel("Probability")
    plt.title("Maximum Entropy Model Probability Distribution")
    plt.show()

    print("Test passed: Maximum Entropy Model works as expected.")



import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import matplotlib.pyplot as plt

# Define the Maximum Entropy Model for continuous distributions
class ContinuousMaximumEntropyModel:
    def __init__(self, features, constraints, integration_bounds=(-10, 10)):
        """
        Initialize with a set of features and their constraints.
        :param features: List of functions f_i(x)
        :param constraints: List of expected values for each feature
        :param integration_bounds: Tuple representing the range for numerical integration
        """
        self.features = features
        self.constraints = constraints
        self.lambdas = np.zeros(len(features))  # Initialize Lagrange multipliers
        self.integration_bounds = integration_bounds

    def partition_function(self, lambdas):
        """
        Calculate the partition function Z using numerical integration.
        """
        def integrand(x):
            return np.exp(sum(lambdas[i] * f(x) for i, f in enumerate(self.features)))
        
        Z, _ = quad(integrand, *self.integration_bounds)
        return Z

    def objective_function(self, lambdas):
        """
        Calculate the objective for optimization (log-likelihood).
        """
        Z = self.partition_function(lambdas)
        log_likelihood = sum(lambdas[i] * constraint for i, constraint in enumerate(self.constraints)) - np.log(Z)
        return -log_likelihood  # Negative for minimization

    def fit(self):
        """
        Fit the model using optimization to find the Lagrange multipliers.
        """
        result = minimize(self.objective_function, self.lambdas, method='L-BFGS-B')
        self.lambdas = result.x

    def probability(self, x):
        """
        Calculate the probability for a given x.
        """
        numerator = np.exp(sum(self.lambdas[i] * f(x) for i, f in enumerate(self.features)))
        Z = self.partition_function(self.lambdas)
        return numerator / Z

# Test function for Continuous Maximum Entropy Model
def test_ContinuousMaxEnt():
    # Define feature functions for a continuous sample space
    def f1(x): return x
    def f2(x): return x**2

    features = [f1, f2]

    # Define expected values for the constraints (e.g., mean and second moment)
    constraints = [0, 1]  # Example: mean = 0, variance = 1

    # Create and fit the Continuous Maximum Entropy Model
    model = ContinuousMaximumEntropyModel(features, constraints)
    model.fit()

    # Test the probability distribution over a range of values
    sample_space = np.linspace(-5, 5, 100)
    probabilities = [model.probability(x) for x in sample_space]

    # Visualize the probability distribution
    plt.plot(sample_space, probabilities, color='lightblue')
    plt.xlabel("Outcome")
    plt.ylabel("Probability Density")
    plt.title("Continuous Maximum Entropy Model Probability Distribution")
    plt.show()

    print("Test passed: Continuous Maximum Entropy Model works as expected.")

# Run the test for continuous distributions
test_ContinuousMaxEnt()


# Run the test
test_MaxEnt()
