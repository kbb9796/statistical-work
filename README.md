# statistical-work

This repository contains: (1) code and visuals for statistical projects I work on in my free time, (2) academic papers explaining computational neuroscience projects I plan to complete at the Miri Lab, and (3) an explanation of some mathematical projects I completed during undergrad. I intend for this repository to give the reader an idea of what mathematical and statistical ideas I enjoy working on and would like to explore further. The Python notebooks are contained in the folders above. 

In my free time, I have explored two statistical approaches in particular: Markov Chain Monte Carlo (MCMC) and latent variable methods. Several Python notebooks above apply MCMC to statistical mechanics, cryptography, and a basic NP-hard problem. Specifically, I apply the Metropolis-Hastings algorithm to calculate otherwise intractable quantities of interest in the Ising model. Similarly, I decoded encrypted messages using MCMC. And lastly, I used simulated annealing to approximate solutions to the famous traveling salesman problem. As for the latent variable methods, I wrote two notebooks that briefly describe the data augmentation algorithm and apply it in two different settings. First, I use data augmentation to simplify the form of an analytically messy posterior distribution in order to sample it more efficiently. Then, with a different data set, I augment right-censored regression data in order to incorporate as much information as possible into the posterior distribution. Those examples are adapted from Tanner, 1996. 

With regards to my interests at the Miri Lab, there are three statistical projects I would like to complete. All of them fall under the broad umbrella of state space models, which incorporate both MCMC and latent variable methods. I briefly explain the general ideas here, and linked academic papers that explain the mathematics below. First, I want to fit a [Poisson Linear Dynamical System (PLDS) model](https://papers.nips.cc/paper_files/paper/2011/file/7143d7fbadfa4693b9eec507d9d37443-Paper.pdf) to high-dimensional neural timeseries data using an expectation-maximization (EM) algorithm. The PLDS model posits that some low-dimensional latent variables, evolving according to linear dynamics, dictate the single-trial spiking activity of neural populations. These latent trajectories should tell us something about decision-making in the brain in the context of our experimental paradigm. Next, I want to fit a [Kalman filter](https://ani.stat.fsu.edu/~wwu/papers/WuNC06.pdf) to our neural and kinematics data, also using an expectation-maximization algorithm. Specifically, the Kalman filter will take neural activity as input to predict the position, velocity, and acceleration of mouse forelimbs. A Kalman filter can explain how kinematic parameters are encoded in neural signals. And lastly, I would like to fit a Hidden Markov Model (HMM) to classify different states of activity using our neural data. This will also help us understand how neural signals encode kinematic parameters. 

Aditionally, I completed several mathematical projects during undergrad. I completed my favorite project with Dr. Ursula Porod (Associate Chair of the Mathematics Department) on rates of convergence for Markov chains. During the independent study, we explored both probabilistic and spectral methods for determining rates of convergence for Markov chains, and specifically random walks on permutation groups. In the end, I could understand the methods used to determine that it takes 7-8 shuffles to sufficiently mix a deck of 52 cards. I also conducted a study in the math department's Undergraduate Directed Reading Program (UDRP) on ergodic and measure theory. Together with my mentor, we proved major theorems in ergodic theory. In the end, I explained the isomorphic structure between the "doubling map" and binary expansions to the some members of the math department. And lastly, I completed another project in the UDRP where we used Brownian motion to solve the Dirichlet problem. I presented the solutions and simulations to some members of the math department. 

Hopefully, the reader has gained an understanding of the statistical and mathematical work I enjoy and wish to explore further. I understand some of these problems/projects are well studied and there are probably similar resources online, however, I thought it would be useful to write programs solving these problems myself to demonstrate my interest in these topics. Also, it may be important to note that this repository does not encompass all of my statistical interests. There are many more statistical problems I enjoy working on. These are just a few!