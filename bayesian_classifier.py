import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Dataset
X = np.array([1.0, 2.0, 1.5, 2.5, 3.0])
Y = np.array([0, 0, 1, 1, 1])

# Prior probabilities
prior_y0 = np.sum(Y == 0) / len(Y)
prior_y1 = np.sum(Y == 1) / len(Y)

# Likelihood parameters
mu_0, sigma_0 = np.mean(X[Y == 0]), np.std(X[Y == 0])
mu_1, sigma_1 = np.mean(X[Y == 1]), np.std(X[Y == 1])

# Define the likelihood functions for each class
def likelihood(x, mu, sigma):
    return norm.pdf(x, mu, sigma)

# Posterior calculation for a given X
def posterior(x):
    # Likelihoods
    p_x_given_y0 = likelihood(x, mu_0, sigma_0)
    p_x_given_y1 = likelihood(x, mu_1, sigma_1)
    
    # Marginal probability P(X)
    p_x = p_x_given_y0 * prior_y0 + p_x_given_y1 * prior_y1
    
    # Posterior probabilities
    p_y0_given_x = (p_x_given_y0 * prior_y0) / p_x
    p_y1_given_x = (p_x_given_y1 * prior_y1) / p_x
    
    return p_y0_given_x, p_y1_given_x

# Test points to evaluate and visualize the posterior
test_points = np.linspace(0, 4, 100)
posterior_y0 = []
posterior_y1 = []

for x in test_points:
    p_y0, p_y1 = posterior(x)
    posterior_y0.append(p_y0)
    posterior_y1.append(p_y1)

# Plotting the posterior distribution over classes
plt.plot(test_points, posterior_y0, label="P(Y=0 | X=x)")
plt.plot(test_points, posterior_y1, label="P(Y=1 | X=x)")
plt.xlabel("Feature X")
plt.ylabel("Posterior Probability")
plt.title("Posterior Distribution over Classes")
plt.legend()
plt.show()
