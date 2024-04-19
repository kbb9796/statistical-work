## Overview

This repository contains: (1) code and visuals implementing statistical methods, (2) brief explanations of future computational neuroscience projects, and (3) past undergraduate math projects. This repository should give the reader an idea of what math I enjoy and would like to explore further. 

## Personal statistical projects

Several of the Python notebooks implement Markov Chain Monte Carlo (MCMC) methods to sample otherwise intractable probability distributions, allowing us to estimate quantities of interest. Specifically, they apply MCMC to [statistical mechanics](ising_model_mcmc/README.md), cryptography, and basic NP-hard problems. The other notebooks implement latent variable methods. 

In particular, I use sequential importance resampling (SIR) to recover the latent dynamics of a toy state-space model for neural spike trains. The others explore two different purposes of the data augmentation algorithm (Tanner, 1996). First, I simplify the form of an analytically complex posterior distribution. Second, having right-censored data, I leverage data augmentation to better estimate a posterior distribution of linear regression parameters. 

## Potential computational neuroscience projects

As for my interests at the [Miri Lab](https://www.mirilab.org), there are many questions I would like to investigate, but the following two interest me the most. Both of them fall under the umbrella of state space models and incorporate some of the aforementioned methods. First, I want to fit a [Poisson Linear Dynamical System (PLDS) model](https://papers.nips.cc/paper_files/paper/2011/file/7143d7fbadfa4693b9eec507d9d37443-Paper.pdf) to our high-dimensional neural timeseries data. Uncovering single trial latent trajectories in our experimental paradigm should tell us something about decision making in the brain. I would also like to determine whether [diffusion-to-bound or switching models](https://www.cambridge.org/core/books/abs/advanced-state-space-methods-for-neural-and-clinical-data/estimating-state-and-parameters-in-state-space-models-of-spike-trains/FAB8634C2790F3461E3E86BB632EAE6F) best describe our neural data using Bayesian model comparison techniques. Both models produce similar trial averaged results, yet the two models differ immensely. Which model supports the data better?

## Undergraduate math projects

Aditionally, I completed several mathematical projects during undergrad. I completed my favorite project with Dr. Ursula Porod (Associate Chair of the Mathematics Department) on rates of convergence for Markov chains. During the independent study, we explored probabilistic and spectral methods for determining rates of convergence for Markov chains, specifically, random walks on permutation groups. In the Undergraduate Directed Reading Program (UDRP), my mentor and I proved major theorems in ergodic theory, and uncovered the isomorphic structure between the "doubling map" and binary expansions. Lastly, in another UDRP project, my mentor and I used Brownian motion to solve the Dirichlet problem.

Hopefully, the reader has gained an understanding of the statistical and mathematical work that interests me. I understand some of these problems are well studied, and there may be similar resources out there. However, it was a useful exercise to solve these problems and implement the programs myself. Also, note that this repository contains a few, but not all, of my statistical interests. 