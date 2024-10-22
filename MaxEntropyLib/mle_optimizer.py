from max_entropy_base import MaxEntropyBase 
import torch 
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

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

    def _log_likelihood(self):
        """
        Compute negative log-likelihood over the dataset.

        Returns:
        - torch.Tensor: Negative log-likelihood scalar.
        """
        energies = []
        for seq in self.msa_onehot:
            energy = self._energy(seq)
            energies.append(energy)

        energies = torch.stack(energies)  # Shape: (n_sequences,)

        # Approximate partition function (assuming equal probability for observed sequences)
        logZ = torch.logsumexp(-energies, dim=0) - torch.log(torch.tensor(len(energies), dtype=torch.float32, device=self.device))

        neg_log_likelihood = torch.mean(energies) + logZ

        # Regularization
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