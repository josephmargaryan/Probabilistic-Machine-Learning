from max_entropy_base import MaxEntropyBase 
import torch 
from tqdm import tqdm 
import matplotlib.pyplot as plt
import numpy as np

class EMOptimizer(MaxEntropyBase):
    def __init__(self, msa_data, reg_lambda=0.01, use_gpu=False):
        super().__init__(msa_data, reg_lambda, use_gpu)
        self.h = torch.zeros((self.n_positions, self.n_amino_acids), device=self.device)
        self.J = torch.zeros((self.n_positions, self.n_positions, self.n_amino_acids, self.n_amino_acids), device=self.device)

        # Initialize empirical probabilities
        self.compute_empirical_probabilities()

    def fit(self, max_iter=10):
        """
        Fit the model using the EM algorithm.

        Parameters:
        - max_iter (int): Maximum number of EM iterations.
        """
        self.loss_history = []
        for epoch in tqdm(range(max_iter), desc="EM Optimization"):
            # E-step: Compute expected sufficient statistics
            # Placeholder for E-step implementation
            expected_frequencies = self.fi  # Simplification

            # M-step: Maximize parameters given expected statistics
            # Update h and J based on expected frequencies
            self.h = torch.log(expected_frequencies + 1e-8)
            self.J = torch.zeros_like(self.J)  # Simplification

            # Compute loss (negative log-likelihood)
            loss = -torch.sum(self.h * self.fi)
            self.loss_history.append(loss.item())

    def predict(self):
        """
        Return the estimated parameters.

        Returns:
        - torch.Tensor: Biases h.
        - torch.Tensor: Couplings J.
        """
        return self.h.detach().cpu().numpy(), self.J.detach().cpu().numpy()

    def visualize(self):
        """
        Visualize convergence of the EM algorithm.
        """
        plt.figure(figsize=(10, 4))
        plt.plot(self.loss_history)
        plt.xlabel('Iteration')
        plt.ylabel('Negative Log-Likelihood')
        plt.title('EM Optimization Progress')
        plt.show()


def test():
    msa_data = np.random.randint(0, 20, (100, 50))  # 100 sequences, 50 positions
    # Initialize and run EM optimizer
    em_optimizer = EMOptimizer(msa_data, reg_lambda=0.01, use_gpu=False)
    em_optimizer.fit(max_iter=10)
    em_optimizer.visualize()

    # Predict parameters
    h_em, J_em = em_optimizer.predict()
    print(h_em, J_em)

if __name__ == "__main__":
    test()