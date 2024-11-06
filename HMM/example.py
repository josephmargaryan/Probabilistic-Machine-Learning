import numpy as np
from hmmlearn import hmm

# Initialize the HMM model
num_states = 2  # "Rainy" and "Sunny"
model = hmm.MultinomialHMM(n_components=num_states, n_iter=100, random_state=42)

# Set initial state probabilities
model.startprob_ = np.array([0.6, 0.4])

# Set transition probabilities
model.transmat_ = np.array([
    [0.7, 0.3],
    [0.4, 0.6]
])

# Set emission probabilities
model.emissionprob_ = np.array([
    [0.1, 0.4, 0.5],  # Rainy
    [0.6, 0.3, 0.1]   # Sunny
])

# Observation sequence (encoded): "Walk", "Shop", "Clean"
obs_seq = np.array([[0], [1], [2], [0], [1], [0], [2]])  # Shape (num_samples, 1)

# Fit the model using the Baum-Welch algorithm (optional step to refine parameters)
model.fit(obs_seq)

# Print the updated model parameters after fitting
print("Updated start probabilities:", model.startprob_)
print("Updated transition matrix:", model.transmat_)
print("Updated emission probabilities:", model.emissionprob_)

# Predict the hidden states for the observation sequence
hidden_states = model.predict(obs_seq)
print("Predicted hidden states:", hidden_states)

# Decode the observation sequence with the Viterbi algorithm
logprob, state_sequence = model.decode(obs_seq, algorithm="viterbi")
print("Log probability of the best state sequence:", logprob)
print("Best state sequence:", state_sequence)

# Compute forward probabilities
log_likelihoods = model._compute_log_likelihood(obs_seq)
fwd_lattice = model._do_forward_pass(log_likelihoods)
print("Forward probabilities:\n", np.exp(fwd_lattice))

# Compute backward probabilities
bwd_lattice = model._do_backward_pass(log_likelihoods)
print("Backward probabilities:\n", np.exp(bwd_lattice))

# Compute posterior probabilities (gamma) for each state at each time step
posterior_probs = model.predict_proba(obs_seq)
print("Posterior probabilities:\n", posterior_probs)

# Generate new data from the model
generated_obs, generated_states = model.sample(n_samples=10)
print("Generated observations:", generated_obs.flatten())
print("Generated hidden states:", generated_states)

