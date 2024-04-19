## Overview

This repository contains: (1) code and visuals implementing statistical methods I find interesting, (2) brief explanations of planned computational neuroscience projects, and (3) brief descriptions of past undergraduate math projects. This repository should give the reader an idea of what math I enjoy and would like to explore further. 

## Personal statistical projects

Several Python notebooks implement Markov Chain Monte Carlo (MCMC) methods. MCMC enables us to sample otherwise intractable probability distributions and estimate quantities of interest. Specifically, I apply MCMC to [statistical mechanics](ising_model_mcmc/README.md), cryptography, and basic NP-hard problems. The other notebooks implement latent variable methods. 

In particular, one notebook employs sequential importance resampling (SIR) to recover the latent dynamics of a toy state-space model. The others explore the data augmentation algorithm (Tanner, 1996). One application simplifies the form of an analytically complex posterior distribution, and the other better estimates a posterior of linear regression parameters given right-censored data.

## Potential computational neuroscience projects

At the [Miri Lab](https://www.mirilab.org), I would like to investigate state-space models for neural computation in "self-paced" decision making. Namely, with high dimensional neural timeseries data, I want to uncover latent dynamics using a [Poisson Linear Dynamical System (PLDS) model](https://papers.nips.cc/paper_files/paper/2011/file/7143d7fbadfa4693b9eec507d9d37443-Paper.pdf). Similarly, I would like to compare the [diffusion-to-bound model to the switching model](https://www.cambridge.org/core/books/abs/advanced-state-space-methods-for-neural-and-clinical-data/estimating-state-and-parameters-in-state-space-models-of-spike-trains/FAB8634C2790F3461E3E86BB632EAE6F) and determine which model best describes our data.

## Undergraduate math projects

I explored rates of convergence for Markov chains with Dr. Ursula Porod (Associate Chair of Mathematics Department) for a quarter. We studied probabilistic and spectral methods for determining rates of convergence, specifically for random walks on permutation groups. In the Directed Reading Program (DRP), my mentor and I proved major theorems in ergodic theory, and uncovered the isomorphic structure between the "doubling map" and binary expansions. And with another mentor, I used Brownian motion to solve the Dirichlet problem for the heat equation. We coded simulations to find the temperature distribution on the interior of a domain given an initial temperature distribution on its boundary. 

## Please check out the notebooks above!

Hopefully, the reader will gain an understanding of what statistical and mathematical work interests me. I understand some of these problems are well studied, and there may be similar work out there. Nonetheless, solving these problems and writing the programs myself has been a useful exercise. 