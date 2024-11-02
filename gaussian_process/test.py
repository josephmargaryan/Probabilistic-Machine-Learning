import numpy as np
import matplotlib.pyplot as plt
import GPy

# Generate or load financial time series data
# Let's assume we're working with stock prices or returns here
np.random.seed(42)
n_points = 100
X = np.linspace(0, 10, n_points).reshape(-1, 1)
Y = np.sin(X) + np.random.normal(
    0, 0.1, X.shape
)  # Simulated sinusoidal data with noise

# Define the kernel for the GP (RBF kernel is common in finance)
kernel = GPy.kern.RBF(input_dim=1, variance=1.0, lengthscale=1.0)

# Create and train the Gaussian Process model
model = GPy.models.GPRegression(X, Y, kernel)
model.optimize(messages=True)

# Plot the model fit with uncertainty
model.plot(plot_density=True)
plt.title("Gaussian Process Regression on Simulated Data")
plt.xlabel("Time")
plt.ylabel("Price")
plt.show()

# Forecast future points
X_future = np.linspace(10, 12, 10).reshape(-1, 1)
Y_future_mean, Y_future_var = model.predict(X_future)

# Plot historical data, model fit, and future forecasts
plt.figure(figsize=(10, 6))
plt.plot(X, Y, "kx", label="Observed Data")
model.plot(plot_density=True)
plt.plot(X_future, Y_future_mean, "ro", label="Forecasted Mean")
plt.fill_between(
    X_future.flatten(),
    (Y_future_mean - 1.96 * np.sqrt(Y_future_var)).flatten(),
    (Y_future_mean + 1.96 * np.sqrt(Y_future_var)).flatten(),
    color="pink",
    alpha=0.5,
    label="95% Confidence Interval",
)
plt.legend()
plt.title("GP Forecast with Confidence Intervals")
plt.show()

# Define simple GP-based trading strategy
initial_capital = 1000
capital = initial_capital
positions = []
returns = []

for t in range(1, n_points):
    X_train = X[:t]
    Y_train = Y[:t]

    # Re-train model on the expanding window of past data
    model = GPy.models.GPRegression(X_train, Y_train, kernel)
    model.optimize(messages=False)

    # Predict the next step
    X_next = X[t].reshape(-1, 1)
    Y_next_mean, _ = model.predict(X_next)

    # Simple mean-reversion strategy
    if Y_next_mean > Y_train[-1]:  # Predicted positive return
        positions.append("Buy")
        capital += Y_next_mean - Y_train[-1]  # Unrealized gain
    else:
        positions.append("Sell")
        capital += Y_train[-1] - Y_next_mean  # Unrealized loss

    returns.append(capital - initial_capital)

# Plot cumulative returns
plt.plot(range(1, n_points), returns)
plt.xlabel("Time")
plt.ylabel("Cumulative Returns")
plt.title("GP-based Mean Reversion Strategy Returns")
plt.show()
