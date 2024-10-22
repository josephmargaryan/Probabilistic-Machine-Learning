import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class MaxEntropyBase:
    def __init__(self, msa_data, reg_lambda=0.01, use_gpu=False):
        """
        Base class for Maximum Entropy models.

        Parameters:
        - msa_data (np.array): Multiple sequence alignment data (n_sequences, n_positions).
        - reg_lambda (float): Regularization parameter.
        - use_gpu (bool): Whether to use GPU for computation.
        """
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.msa_data = torch.tensor(msa_data, dtype=torch.long, device=self.device)
        self.n_sequences, self.n_positions = msa_data.shape
        self.n_amino_acids = 20  # Standard number of amino acids
        self.reg_lambda = reg_lambda

        # One-hot encoding of MSA data
        self.msa_onehot = self._one_hot_encode(self.msa_data)

        # Parameters to be initialized in subclasses
        self.h = None  # Biases
        self.J = None  # Couplings

    def _one_hot_encode(self, data):
        """
        One-hot encode the MSA data.

        Returns:
        - torch.Tensor: One-hot encoded data (n_sequences, n_positions, n_amino_acids).
        """
        one_hot = torch.zeros((self.n_sequences, self.n_positions, self.n_amino_acids), device=self.device)
        one_hot.scatter_(2, data.unsqueeze(-1), 1)
        return one_hot

    def compute_empirical_probabilities(self):
        """
        Compute empirical marginal and pairwise probabilities from MSA data.
        """
        # Single-site frequencies
        self.fi = torch.mean(self.msa_onehot, dim=0)  # Shape: (n_positions, n_amino_acids)

        # Pairwise frequencies
        self.fij = torch.einsum('sik,sjl->ijkl', self.msa_onehot, self.msa_onehot) / self.n_sequences
        # Shape: (n_positions, n_positions, n_amino_acids, n_amino_acids)

    def visualize_empirical_probabilities(self):
        """
        Visualize empirical marginal probabilities as a heatmap.
        """
        fi_numpy = self.fi.detach().cpu().numpy()
        plt.figure(figsize=(10, 6))
        plt.imshow(fi_numpy.T, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title('Empirical Marginal Probabilities')
        plt.xlabel('Position')
        plt.ylabel('Amino Acid Index')
        plt.show()
