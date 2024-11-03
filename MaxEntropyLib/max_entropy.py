import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


class MaxEntropyModel:
    def __init__(self, data, n_categories, reg_lambda=0.01, use_gpu=False):
        """
        Maximum Entropy model with adjustable dimensions for domain flexibility.

        Parameters:
        - data (np.array): Input data (n_samples, n_features).
        - n_categories (int): Number of unique categories per feature (e.g., 20 for amino acids).
        - reg_lambda (float): Regularization parameter.
        - use_gpu (bool): Whether to use GPU for computation.
        """
        self.device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        self.data = torch.tensor(data, dtype=torch.long, device=self.device)
        self.n_samples, self.n_features = data.shape
        self.n_categories = n_categories
        self.reg_lambda = reg_lambda

        # One-hot encoding of data
        self.data_onehot = self._one_hot_encode(self.data)

        # Parameters
        self.h = torch.zeros(
            (self.n_features, self.n_categories), device=self.device, requires_grad=True
        )
        self.J = torch.zeros(
            (self.n_features, self.n_features, self.n_categories, self.n_categories),
            device=self.device,
            requires_grad=True,
        )

        # Enforce diagonal of J to be zero for non-self interactions
        self.J.data[torch.arange(self.n_features), torch.arange(self.n_features)] = 0

        # Calculate empirical frequencies
        self.compute_empirical_probabilities()

    def _one_hot_encode(self, data):
        """
        One-hot encode the input data.

        Returns:
        - torch.Tensor: One-hot encoded data (n_samples, n_features, n_categories).
        """
        one_hot = torch.zeros(
            (self.n_samples, self.n_features, self.n_categories), device=self.device
        )
        one_hot.scatter_(2, data.unsqueeze(-1), 1)
        return one_hot

    def compute_empirical_probabilities(self):
        """
        Compute empirical marginal and pairwise probabilities from data.
        """
        # Single-site frequencies
        self.fi = torch.mean(
            self.data_onehot, dim=0
        )  # Shape: (n_features, n_categories)

        # Pairwise frequencies
        self.fij = (
            torch.einsum("sik,sjl->ijkl", self.data_onehot, self.data_onehot)
            / self.n_samples
        )

    def _compute_pseudo_likelihood(self, seq_onehot):
        """
        Compute pseudo-likelihood approximation for a given sequence.
        Parameters:
        - seq_onehot (torch.Tensor): One-hot encoded sequence.
        Returns:
        - torch.Tensor: Approximate likelihood.
        """
        h_terms = torch.sum(self.h * seq_onehot)

        # Calculate pairwise interactions while excluding self-interactions
        J_terms = torch.einsum("ia,jb,ijab->", seq_onehot, seq_onehot, self.J)
        energy = -(h_terms + J_terms)
        return energy

    def _log_likelihood(self):
        """
        Compute approximate log-likelihood using pseudo-likelihood.

        Returns:
        - torch.Tensor: Negative log-likelihood.
        """
        energies = [self._compute_pseudo_likelihood(seq) for seq in self.data_onehot]
        energies = torch.stack(energies)

        # Approximate partition function Z with mean energies
        logZ = torch.mean(energies)
        neg_log_likelihood = torch.mean(energies) - logZ

        # Regularization term for L2 penalty on parameters
        reg = self.reg_lambda * (torch.sum(self.h**2) + torch.sum(self.J**2))
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
        pbar = tqdm(range(max_iter), desc="Training MaxEntropyModel")

        for epoch in pbar:
            optimizer.zero_grad()
            loss = self._log_likelihood()
            loss.backward()
            optimizer.step()

            self.loss_history.append(loss.item())
            pbar.set_postfix({"Loss": loss.item()})

    def compute_direct_information(self):
        """
        Compute Direct Information (DI) based on the model parameters.

        Returns:
        - np.array: DI matrix.
        """
        # Compute direct probabilities
        P_dir = torch.exp(self.J)
        P_dir = P_dir / torch.sum(P_dir, dim=(2, 3), keepdim=True)  # Normalize

        fi = self.fi.unsqueeze(1).unsqueeze(3)  # Reshape fi
        fj = self.fi.unsqueeze(0).unsqueeze(2)  # Reshape fj

        # Compute Direct Information (DI) from normalized probabilities
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
        plt.xlabel("Iteration")
        plt.ylabel("Negative Log-Likelihood")
        plt.title("Training Progress")
        plt.show()

        # Visualize coupling matrix
        coupling_matrix = (
            torch.sum(torch.abs(self.J), dim=(2, 3)).detach().cpu().numpy()
        )
        plt.figure(figsize=(6, 5))
        plt.imshow(coupling_matrix, cmap="viridis")
        plt.colorbar()
        plt.title("Coupling Matrix")
        plt.xlabel("Feature i")
        plt.ylabel("Feature j")
        plt.show()


# Example test function for domain flexibility
def test():
    # Generate synthetic data for testing
    np.random.seed(42)
    test_data = np.random.randint(
        0, 20, (100, 50)
    )  # 100 samples, 50 features, 20 categories

    # Initialize and train the model
    model = MaxEntropyModel(test_data, n_categories=20, reg_lambda=0.01, use_gpu=False)
    model.fit(max_iter=50, lr=0.05)
    model.visualize()

    # Predict coupling matrix and compute DI
    coupling_matrix = model.predict()
    DI_matrix = model.compute_direct_information()
    print("Coupling Matrix:\n", coupling_matrix)
    print("Direct Information Matrix:\n", DI_matrix)

    # Visualize DI matrix
    plt.figure(figsize=(6, 5))
    plt.imshow(DI_matrix, cmap="hot")
    plt.colorbar()
    plt.title("Direct Information Matrix")
    plt.xlabel("Feature i")
    plt.ylabel("Feature j")
    plt.show()


if __name__ == "__main__":
    test()
