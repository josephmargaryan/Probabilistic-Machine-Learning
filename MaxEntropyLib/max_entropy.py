import torch 
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

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

class MLEOptimizer(MaxEntropyBase):
    def __init__(self, msa_data, reg_lambda=0.01, use_gpu=False):
        super().__init__(msa_data, reg_lambda, use_gpu)
        self.h = torch.zeros((self.n_positions, self.n_amino_acids), device=self.device, requires_grad=True)
        self.J = torch.zeros((self.n_positions, self.n_positions, self.n_amino_acids, self.n_amino_acids), device=self.device, requires_grad=True)

        # Initialize empirical probabilities
        self.compute_empirical_probabilities()

    def _energy(self, seq_onehot):
        """
        Compute energy for a given sequence.

        Parameters:
        - seq_onehot (torch.Tensor): One-hot encoded sequence (n_positions, n_amino_acids).

        Returns:
        - torch.Tensor: Energy scalar.
        """
        # Single-site terms
        h_terms = torch.sum(self.h * seq_onehot)

        # Pairwise terms using torch.einsum
        J_terms = torch.einsum('ia,jb,ijab->', seq_onehot, seq_onehot, self.J)

        return - (h_terms + J_terms)
    
    def _compute_hamiltonian(self, seq_onehot):
        """
        Compute Hamiltonian (energy) for a given sequence.
        """
        # Single-site terms
        h_terms = torch.sum(self.h * seq_onehot)

        # Pairwise terms using torch.einsum
        J_terms = torch.einsum('ia,jb,ijab->', seq_onehot, seq_onehot, self.J)

        return -(h_terms + J_terms)


    def _log_likelihood(self):
        energies = [self._compute_hamiltonian(seq) for seq in self.msa_onehot]
        energies = torch.stack(energies)
        logZ = torch.logsumexp(-energies, dim=0) - torch.log(torch.tensor(len(energies), dtype=torch.float32, device=self.device))
        neg_log_likelihood = torch.mean(energies) + logZ
        reg = self.reg_lambda * (torch.sum(self.h ** 2) + torch.sum(self.J ** 2))
        return neg_log_likelihood + reg


    def fit(self, max_iter=100, lr=0.1):
        """
        Fit the model using gradient descent.

        Parameters:
        - max_iter (int): Maximum number of iterations.
        - lr (float): Learning rate.
        """
        optimizer = torch.optim.Adam([self.h, self.J], lr=lr)
        self.loss_history = []
        pbar = tqdm(range(max_iter), desc="Training MLEOptimizer")

        for epoch in pbar:
            optimizer.zero_grad()
            loss = self._log_likelihood()
            loss.backward()
            optimizer.step()

            self.loss_history.append(loss.item())
            pbar.set_postfix({'Loss': loss.item()})

    def compute_direct_information(self):
        """
        Compute Direct Information (DI) based on the model parameters.

        Returns:
        - np.array: DI matrix.
        """
        # Compute the direct probabilities (P_dir)
        P_dir = torch.exp(self.J)
        P_dir = P_dir / torch.sum(P_dir, dim=(2, 3), keepdim=True)  # Normalize over amino acids

        fi = self.fi.unsqueeze(1).unsqueeze(3)  # Shape: (n_positions, 1, n_amino_acids, 1)
        fj = self.fi.unsqueeze(0).unsqueeze(2)  # Shape: (1, n_positions, 1, n_amino_acids)

        # Compute DI
        DI = torch.sum(P_dir * torch.log((P_dir + 1e-8) / (fi * fj + 1e-8)), dim=(2, 3))
        return DI.detach().cpu().numpy()

    def predict(self):
        """
        Predict the coupling matrix e_{ij}(A_i, A_j).

        Returns:
        - np.array: Coupling matrix.
        """
        return self.J.detach().cpu().numpy()

    def visualize(self):
        """
        Visualize training progress and coupling matrix.
        """
        # Plot loss over iterations
        plt.figure(figsize=(10, 4))
        plt.plot(self.loss_history)
        plt.xlabel('Iteration')
        plt.ylabel('Negative Log-Likelihood')
        plt.title('Training Progress')
        plt.show()

        # Visualize coupling matrix
        coupling_matrix = torch.sum(torch.abs(self.J), dim=(2, 3)).detach().cpu().numpy()
        plt.figure(figsize=(6, 5))
        plt.imshow(coupling_matrix, cmap='viridis')
        plt.colorbar()
        plt.title('Coupling Matrix')
        plt.xlabel('Position i')
        plt.ylabel('Position j')
        plt.show()



def test():
    # Generate synthetic MSA data
    np.random.seed(42)
    msa_data = np.random.randint(0, 20, (100, 50))  # 100 sequences, 50 positions

    # Initialize and train the MLE optimizer
    mle_optimizer = MLEOptimizer(msa_data, reg_lambda=0.01, use_gpu=False)
    mle_optimizer.fit(max_iter=50, lr=0.05)
    mle_optimizer.visualize()

    # Predict coupling matrix and compute DI
    coupling_matrix = mle_optimizer.predict()
    DI_matrix = mle_optimizer.compute_direct_information()
    print(coupling_matrix)
    print(DI_matrix)
    # Visualize DI matrix
    plt.figure(figsize=(6, 5))
    plt.imshow(DI_matrix, cmap='hot')
    plt.colorbar()
    plt.title('Direct Information Matrix')
    plt.xlabel('Position i')
    plt.ylabel('Position j')
    plt.show()

if __name__ == "__main__":
    test()