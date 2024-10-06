# Bayesian Deep Learning with Circulant Weights

This project explores Bayesian deep learning using circulant matrices, Fast Fourier Transform (FFT), and efficient MCMC methods (Hamiltonian Monte Carlo with Energy Conserving Subsampling - HMCECS). The goal is to enhance deep learning models with uncertainty quantification and reduce dimensionality using circulant weight matrices.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [How to Run](#how-to-run)
- [Theory Behind the Model](#theory-behind-the-model)
- [Results and Evaluation](#results-and-evaluation)
- [References](#references)

## Introduction

In traditional deep learning, models often lack uncertainty estimation and can be overconfident in their predictions. Bayesian deep learning solves this problem by using probability distributions over model parameters, providing a way to quantify uncertainty.

This project leverages:
- **Circulant Weight Matrices** to reduce the dimensionality of deep learning models by exploiting their structure.
- **Fast Fourier Transform (FFT)** for efficient matrix multiplication without explicitly constructing the circulant matrix.
- **HMCECS (Hamiltonian Monte Carlo with Energy Conserving Subsampling)** for handling large datasets with MCMC, enabling efficient mini-batching.

## Project Structure

- `bayesian_fft_mcmc.py`: The main Python script containing the implementation of the Bayesian deep learning model, MCMC sampling, and evaluation metrics.
- `README.md`: This documentation file.
  
## Dependencies

This project requires the following Python libraries:
- `numpy`
- `jax`
- `numpyro`
- `matplotlib`
- `sklearn`

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
