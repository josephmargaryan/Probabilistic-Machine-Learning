# Bayesian Deep Learning with Circulant Weights

This project explores Bayesian deep learning using circulant matrices, Fast Fourier Transform (FFT), and efficient MCMC methods (Hamiltonian Monte Carlo with Energy Conserving Subsampling - HMCECS). The goal is to enhance deep learning models with uncertainty quantification and reduce dimensionality using circulant weight matrices.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [How to Run](#how-to-run)
- [Theory Behind the Model](#theory-behind-the-model)
- [References](#references)

## Introduction

In traditional deep learning, models often need more certainty estimation and can be overconfident in their predictions. Bayesian deep learning solves this problem by using probability distributions over model parameters, providing a way to quantify uncertainty.

This project leverages:
- **Circulant Weight Matrices** to reduce the dimensionality of deep learning models by exploiting their structure.
- **Fast Fourier Transform (FFT)** for efficient matrix multiplication without explicitly constructing the circulant matrix.
- **HMCECS (Hamiltonian Monte Carlo with Energy Conserving Subsampling)** for handling large datasets with MCMC, enabling efficient mini-batching.

## Project Structure

- `bayesian_fft_mcmc.py`: The main Python script containing implementing the Bayesian deep learning model, MCMC sampling, and evaluation metrics.
- `README.md`: This documentation file.
  
## Dependencies

This project requires the following Python libraries:
- `numpy`
- `jax`
- `numpyro`
- `matplotlib`
- `sklearn`

## Theory Behind the Model
### Bayesian Neural Networks
A Bayesian neural network (BNN) estimates a probability distribution over its weights rather than a single point estimate, providing a principled way to quantify uncertainty in predictions.
### Circulant Matrices and FFT
Circulant matrices allow for efficient computation of matrix-vector products using the Fast Fourier Transform (FFT). Instead of multiplying matrices directly, which can be computationally expensive, FFT reduces the complexity by transforming the matrix multiplication into a pointwise product in the Fourier domain.
### HMCECS (Hamiltonian Monte Carlo with Energy Conserving Subsampling)
HMCECS is a Hamiltonian Monte Carlo (HMC) variant designed to efficiently handle large datasets' mini-batches. It estimates the likelihood of the entire dataset using only a small subset (mini-batch) of data while conserving energy in the Hamiltonian system.
### Model Training
The model is trained using a combination of:

- **Subsampling Monte Carlo (HMCECS) to draw posterior samples overweights.**
- **SVI (Stochastic Variational Inference) to initialize the parameters for the MCMC method.**
## References
- **Numpyro: https://num.pyro.ai/en/stable/**
- **Fast Fourier Transform: https://en.wikipedia.org/wiki/Fast_Fourier_transform**
- **HMCECS: Hamiltonian Monte Carlo with Energy Conserving Subsampling, Dang, et al., (2019)**

