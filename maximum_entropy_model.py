import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns

# Synthetic MSA sequences encoded as numerical values
# A -> 0, B -> 1
msa_sequences = np.array([
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, 1],
    [0, 0, 0]
])

num_sequences, sequence_length = msa_sequences.shape
num_states = 2  # A and B


def compute_marginal_frequencies(msa, position):
    counts = np.bincount(msa[:, position], minlength=num_states)
    frequencies = counts / msa.shape[0]
    return frequencies

f_i = [compute_marginal_frequencies(msa_sequences, i) for i in range(sequence_length)]


def compute_joint_frequencies(msa, pos_i, pos_j):
    joint_counts = np.zeros((num_states, num_states))
    for seq in msa:
        ai = seq[pos_i]
        aj = seq[pos_j]
        joint_counts[ai, aj] += 1
    joint_frequencies = joint_counts / msa.shape[0]
    return joint_frequencies

f_ij = {}
for i in range(sequence_length):
    for j in range(i+1, sequence_length):
        f_ij[(i, j)] = compute_joint_frequencies(msa_sequences, i, j)


def objective(params, f_i, f_ij):
    # Unpack parameters
    e_params = {}
    idx = 0
    for key in f_ij.keys():
        e_params[key] = params[idx:idx+num_states**2].reshape((num_states, num_states))
        idx += num_states**2
    
    # Compute model joint probabilities
    P_ij_model = {}
    for key in f_ij.keys():
        e = e_params[key]
        e_exp = np.exp(e)
        Z = np.sum(e_exp)
        P_ij_model[key] = e_exp / Z
    
    # Compute the difference between model and empirical joint frequencies
    diff = 0
    for key in f_ij.keys():
        diff += np.sum((P_ij_model[key] - f_ij[key])**2)
    
    return diff


# Initial guess for parameters
initial_params = np.zeros(len(f_ij) * num_states**2)

# Bounds can be set if needed
bounds = [(-5, 5) for _ in initial_params]


result = minimize(objective, initial_params, args=(f_i, f_ij), bounds=bounds, method='L-BFGS-B')

# Extract optimized parameters
optimized_params = result.x

# Compute direct probabilities and DI
e_params = {}
idx = 0
DI = {}
for key in f_ij.keys():
    e = optimized_params[idx:idx+num_states**2].reshape((num_states, num_states))
    idx += num_states**2
    e_exp = np.exp(e)
    Z = np.sum(e_exp)
    P_dir = e_exp / Z
    # Compute DI
    f_i_prod = np.outer(f_i[key[0]], f_i[key[1]])
    DI_value = np.sum(P_dir * np.log((P_dir + 1e-10) / (f_i_prod + 1e-10)))
    DI[key] = DI_value

# Print DI values
for key, value in DI.items():
    print(f"DI between positions {key[0]+1} and {key[1]+1}: {value}")



# Create an empty matrix for DI values
DI_matrix = np.zeros((sequence_length, sequence_length))

# Fill the matrix with DI values
for (i, j), value in DI.items():
    DI_matrix[i, j] = value
    DI_matrix[j, i] = value  # Symmetric matrix

# Plot the heatmap
sns.heatmap(DI_matrix, annot=True, fmt=".2f", cmap='viridis')
plt.title('Direct Information (DI) Heatmap')
plt.xlabel('Residue Position')
plt.ylabel('Residue Position')
plt.show()
