## Overview

This repository contains: (1) code implementing statistical methods I find interesting, (2) brief explanations of potential computational neuroscience projects, and (3) brief descriptions of past undergraduate math projects. This repository should give the reader an idea of what math I enjoy and would like to explore further. 

## Personal statistical projects

Several of the Python notebooks implement Markov Chain Monte Carlo (MCMC) methods. MCMC allows us to sample otherwise intractable probability distributions and consequently estimate quantities of interest. Specifically, they apply MCMC to [statistical mechanics](ising_model_mcmc/README.md), cryptography, and basic NP-hard problems. The other notebooks implement latent variable methods. 

In particular, one notebook uses sequential importance resampling (SIR) to recover the latent dynamics of a toy state-space model for neural spike trains. The others explore two applications of data augmentation (Tanner, 1996)--simplify the form of an analytically complex posterior distribution, and better estimate the posterior of linear regression parameters given right-censored data.

## Potential computational neuroscience projects

There are many questions I would like to investigate at the [Miri Lab](https://www.mirilab.org), but the following two interest me the most. Both of them fall under the umbrella of state space models and incorporate methods described above. I want to fit a [Poisson Linear Dynamical System (PLDS) model](https://papers.nips.cc/paper_files/paper/2011/file/7143d7fbadfa4693b9eec507d9d37443-Paper.pdf) to our high-dimensional neural timeseries data. Uncovering single trial latent dynamics in our experimental paradigm should tell us something about how the brain makes decisions. And using Bayesian model comparisons, I would like to determine whether a [diffusion-to-bound or switching model](https://www.cambridge.org/core/books/abs/advanced-state-space-methods-for-neural-and-clinical-data/estimating-state-and-parameters-in-state-space-models-of-spike-trains/FAB8634C2790F3461E3E86BB632EAE6F) best describes neural computation during decision making.

## Undergraduate math projects

I completed several math projects during undergrad. I explored rates of convergence for Markov chains with Dr. Ursula Porod (Associate Chair of the Mathematics Department). We studied probabilistic and spectral methods for determining rates of convergence, specifically for random walks on permutation groups. In the Directed Reading Program (DRP), my mentor and I proved major theorems in ergodic theory, and uncovered the isomorphic structure between the "doubling map" and binary expansions. And with another mentor, I used Brownian motion to solve the Dirichlet problem. 

## Please check out the notebooks above!

Hopefully, the reader will gain an understanding of what statistical and mathematical work interests me. I understand some of these problems are well studied, and there may be similar resources out there. Nonetheless, it was a useful exercise to solve these problems and implement the programs myself. Please note that this repository contains just a few, and not all, of my interests.