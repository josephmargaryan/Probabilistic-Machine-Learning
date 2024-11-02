import numpy as np
import matplotlib.pyplot as plt


def simulate_brownian_motion(n_steps=1000, T=1, mu=0, sigma=1):
    dt = T / n_steps
    increments = np.random.normal(mu * dt, sigma * np.sqrt(dt), n_steps)
    W = np.cumsum(increments)  # Cumulative sum to get the Brownian path
    W = np.insert(W, 0, 0)  # Start at zero
    return W


# Simulate and plot
n_steps = 1000
W = simulate_brownian_motion(n_steps=n_steps)
plt.plot(np.linspace(0, 1, n_steps + 1), W)
plt.title("Brownian Motion Simulation")
plt.xlabel("Time")
plt.ylabel("W(t)")
plt.show()


def simulate_gbm(S0, mu, sigma, T=1, n_steps=1000):
    dt = T / n_steps
    t = np.linspace(0, T, n_steps + 1)
    W = simulate_brownian_motion(n_steps=n_steps, T=T)
    S = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)
    return S


# Parameters
S0 = 100  # Initial stock price
mu = 0.05  # Expected return
sigma = 0.2  # Volatility
T = 1  # Time in years

# Simulate GBM
S = simulate_gbm(S0, mu, sigma, T)
plt.plot(np.linspace(0, T, len(S)), S)
plt.title("Geometric Brownian Motion Simulation")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.show()


def euler_maruyama(S0, mu, sigma, T=1, n_steps=1000):
    dt = T / n_steps
    S = np.zeros(n_steps + 1)
    S[0] = S0
    for i in range(1, n_steps + 1):
        dW = np.random.normal(0, np.sqrt(dt))
        S[i] = S[i - 1] + mu * S[i - 1] * dt + sigma * S[i - 1] * dW
    return S


# Simulate SDE using Euler-Maruyama
S = euler_maruyama(S0, mu, sigma, T)
plt.plot(np.linspace(0, T, len(S)), S)
plt.title("Euler-Maruyama Simulation of GBM SDE")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.show()


from scipy.stats import norm


def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# Parameters
S = 100  # Current stock price
K = 100  # Strike price
T = 1  # Time to maturity (1 year)
r = 0.05  # Risk-free rate
sigma = 0.2  # Volatility

# Calculate option prices
call_price = black_scholes(S, K, T, r, sigma, option_type="call")
put_price = black_scholes(S, K, T, r, sigma, option_type="put")
print("Call Option Price:", call_price)
print("Put Option Price:", put_price)


def monte_carlo_option_price(
    S0, K, T, r, sigma, n_simulations=10000, option_type="call"
):
    payoffs = []
    for _ in range(n_simulations):
        S_T = simulate_gbm(S0, r, sigma, T)[-1]  # Final price at maturity
        if option_type == "call":
            payoffs.append(max(S_T - K, 0))
        elif option_type == "put":
            payoffs.append(max(K - S_T, 0))
    return np.exp(-r * T) * np.mean(payoffs)  # Discounted average payoff


# Calculate option prices with Monte Carlo
call_mc_price = monte_carlo_option_price(S0, K, T, r, sigma, option_type="call")
put_mc_price = monte_carlo_option_price(S0, K, T, r, sigma, option_type="put")
print("Monte Carlo Call Option Price:", call_mc_price)
print("Monte Carlo Put Option Price:", put_mc_price)
