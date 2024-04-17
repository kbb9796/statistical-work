## Overview

This repository contains: (1) code and visuals for personal statistical projects (2) brief explanations of potential computational neuroscience projects, and (3) brief explanations of undergraduate math projects. I intend for this repository to give the reader an idea of some of the mathematics I enjoy and would like to explore further. 

## Personal statistical projects

Several of the Python notebooks above implement Markov Chain Monte Carlo (MCMC) methods to sample otherwise intractable probability distributions, allowing us to estimate quantities of interest. Specifically, they apply MCMC to statistical mechanics, cryptography, and basic NP-hard problems. The other notebooks implement latent variable methods. Particularly, one notebook implements sequential importance resampling (SIR) to recover the latent dynamics of a toy state-space model for neural spike trains. The others use the data augmentation algorithm (Tanner, 1996). The first data augmentation notebook simplifies the form of an analytically complex posterior distribution allowing us to sample it efficiently. The second implements the algorithm to better estimate the posterior distribution of linear regression parameters in the case of right-censored data. 

## Potential computational neuroscience projects

With regards to my interests at the Miri Lab, there are many projects I would like to complete, but the following two interest me the most. Both of them fall under the broad umbrella of state space models, incorporating MCMC and latent variable methods. First, I want to fit a [Poisson Linear Dynamical System (PLDS) model](https://papers.nips.cc/paper_files/paper/2011/file/7143d7fbadfa4693b9eec507d9d37443-Paper.pdf) to our high-dimensional neural timeseries data. Uncovering single trial latent trajectories in our experimental paradigm should tell us something about decision making in the brain. I would also like to determine whether [diffusion-to-bound or switching models](https://www.cambridge.org/core/books/abs/advanced-state-space-methods-for-neural-and-clinical-data/estimating-state-and-parameters-in-state-space-models-of-spike-trains/FAB8634C2790F3461E3E86BB632EAE6F) best describe our neural data using Bayesian model comparison techniques. Both models produce similar trial averaged results, yet the two models differ immensely. Which model has more evidence?

## Undergraduate math projects

Aditionally, I completed several mathematical projects during undergrad. I completed my favorite project with Dr. Ursula Porod (Associate Chair of the Mathematics Department) on rates of convergence for Markov chains. During the independent study, we explored probabilistic and spectral methods for determining rates of convergence for Markov chains, specifically, random walks on permutation groups. In the Undergraduate Directed Reading Program (UDRP), my mentor and I proved major theorems in ergodic theory, and uncovered the isomorphic structure between the "doubling map" and binary expansions. Lastly, in another UDRP project, my mentor and I used Brownian motion to solve the Dirichlet problem.

Hopefully, the reader has gained an understanding of the statistical and mathematical work that interests me. I understand some of these problems/projects are well studied, and there may be similar resources out there. However, it was a useful exercise to solve these problems and implement the programs myself. Also, note that this repository does not encompass all of my statistical interests. These are just a few!