# The Ising Model

Statistical physicists want to know how magnetic materials behave at certain temperatures. The Ising model represents exactly that. The model consists of a lattice, where each site on the lattice has a positive or negative spin (represented by a +1 or -1). The lattice configurations behave according to a probability distribution. As usual, we would like to compute some expectations over the probability distribution--for example, what is the average number of positive spins at a given temperature? However, even for a small number of lattice sites, this expectation is intractable. There are too many configurations of the lattice to sum over, so we can't analytically calculate these quantities of interest. But we can estimate them using MCMC methods. 

For a given iteration of the algorithm, we propose a new configuration by flipping one of the spins uniformly at random. The new configuration is either accepted or rejected according to the normal rules of the Metropolis-Hastings algorithm. We code an algorithm to sample the state space of configurations at a given temperature, and use that sample to compute several quantities of interest below.


```python
# Import packages

import numpy as np
import random
import matplotlib.pyplot as plt
from array2gif import write_gif
from tqdm import tqdm
```


```python
# Create initial lattice to start

def create_random_lattice(rows, cols, prob = .5):

    n_trials = 1
    lattice = np.random.binomial(n_trials, prob, [rows, cols])
    lattice_idxs = np.where(lattice == 0)
    lattice[lattice_idxs] = -1

    return lattice

# Compute the probability (up to a normalizing constant) of the configuration

def compute_hamiltonian(lattice):

    num_rows = np.shape(lattice)[0]
    num_cols = np.shape(lattice)[1]
    hamiltonian = 0

    for ii_row in range(np.shape(lattice)[0]):

        for jj_col in range(np.shape(lattice)[1]):

            # Only count neighbors down and to the right to avoid double counting 
            # Periodic boundary conditions
            current_spin = lattice[ii_row, jj_col]
            hamiltonian += current_spin * (lattice[(ii_row + 1) % num_rows, jj_col] + lattice[ii_row, (jj_col + 1) % num_cols])

    return -hamiltonian

def compute_unnormalized_prob(lattice, J_over_kT):

    return np.exp(-J_over_kT * compute_hamiltonian(lattice))

def propose_lattice(lattice):

    num_rows = np.shape(lattice)[0]
    num_cols = np.shape(lattice)[1]
    rnd_row = random.sample(list(range(num_rows)), 1)
    rnd_col = random.sample(list(range(num_cols)), 1)
    new_lattice = np.copy(lattice)
    new_lattice[rnd_row, rnd_col] = -1 * new_lattice[rnd_row, rnd_col] 

    return new_lattice

def metropolis_sampling(lattice, num_iterations, J_over_kT):

    u = np.random.uniform(0, 1, num_iterations)

    with tqdm(total = num_iterations) as pbar:

        sample = np.zeros([np.shape(lattice)[0], np.shape(lattice)[1], num_iterations])

        for ii_iteration in range(num_iterations):
            
            candidate_lattice = propose_lattice(lattice)
            acceptance_prob = np.min([1, np.exp(-J_over_kT * (compute_hamiltonian(candidate_lattice) \
                                                         - compute_hamiltonian(lattice)))])

            if u[ii_iteration] < acceptance_prob:

                lattice = candidate_lattice

            sample[:, :, ii_iteration] = lattice
            pbar.update(1)
    
    return sample

# Write function that creates gif from the sample

def create_image(lattice):

    num_rows = np.shape(lattice)[0]
    num_cols = np.shape(lattice)[1]
    # Turn lattice into image with RGB triplet along third dimension
    image = np.zeros([num_rows, num_cols, 3])
    image[lattice == 1] = [0, 0, 0]
    image[lattice == -1] = [255, 255, 255]

    return image

# Create specific heat function and plot as function of temperature

def compute_specific_heat(samples):

    specific_heat = [np.var(samples[key]) / (float(key) ** 2) for key in samples.keys()]

    return specific_heat

# Create magnetism function that calculates magnetism as function of temperature

def compute_magnetism(samples):

    magnetism = [np.mean(np.ravel(samples[key])) for key in samples.keys()]

    return magnetism
        

```

We sample the equilibrium distribution for the Ising model for a low temperature. You can watch the lattice configuration evolve over time below. At cooler temperatures, the lattice shows some structure. Similar spins are clustered together. At higher temperatures, the model becomes less structured, and the spins are more random. 

# Sampling equilibrium distributions at different temperatures

## Low temperature Ising model

![Sample of equilibrium distribution of Ising Model at low temperature](ising_model_lowtemp.gif)

## High temperature Ising model

![Sample of equilibrium distribution of Ising Model at low temperature](ising_model_hightemp.gif)


```python
num_rows = 100
num_cols = 100
initial_lattice = create_random_lattice(num_rows, num_cols)
num_iterations = 50000
J_over_kT = 10
print('Metropolis sampling')
sample = metropolis_sampling(initial_lattice, num_iterations, J_over_kT)

# Make sure gif happens fast enough

frequency = 500
dataset_for_gif = []

for ii_lattice in range(int(num_iterations / frequency)):

    dataset_for_gif.append(np.array(create_image(sample[:, :, ii_lattice * frequency])))

print('Writing samples to gif')
write_gif(dataset_for_gif, 'ising_model_lowtemp.gif', fps = 5000)

num_rows = 100
num_cols = 100
initial_lattice = np.ones([num_rows, num_cols])
num_iterations = 50000
J_over_kT = .25
print('Metropolis sampling')
sample = metropolis_sampling(initial_lattice, num_iterations, J_over_kT)

# Make sure gif happens fast enough

frequency = 500
dataset_for_gif = []

for ii_lattice in range(int(num_iterations / frequency)):

    dataset_for_gif.append(np.array(create_image(sample[:, :, ii_lattice * frequency])))

print('Writing samples to gif')
write_gif(dataset_for_gif, 'ising_model_hightemp.gif', fps = 5000)

```

      0%|          | 1/50000 [00:00<1:38:19,  8.47it/s]

    Metropolis sampling


    100%|██████████| 50000/50000 [16:59<00:00, 49.05it/s]


    Writing samples to gif


      0%|          | 4/50000 [00:00<20:57, 39.76it/s]

    Metropolis sampling


    100%|██████████| 50000/50000 [17:15<00:00, 48.28it/s] 


    Writing samples to gif


We calculate the specific heat and total magnetism of the Ising model for a lattice of size 20 x 20 at different temperatures below. 


```python
# Calculating specific heat as a function of temperature

num_rows = 20
num_cols = 20
num_iterations = 10000
J_over_kT = np.arange(0, 20, 3)
samples = {}

for ii_temp in J_over_kT:

    print('Metropolis sampling temp: ', ii_temp)
    initial_lattice = create_random_lattice(num_rows, num_cols)
    samples[str(ii_temp)] = metropolis_sampling(initial_lattice, num_iterations, ii_temp)

    # Plot specific heat as function of temperature 

specific_heat = compute_specific_heat(samples)
plt.plot(J_over_kT, specific_heat)
plt.title('Specific Heat over Temperature')
plt.xlabel('Temperature')
plt.ylabel('Specific Heat')
plt.show()
```

      1%|          | 58/10000 [00:00<00:17, 572.61it/s]

    Metropolis sampling temp:  0


    100%|██████████| 10000/10000 [00:12<00:00, 831.16it/s]
      1%|          | 87/10000 [00:00<00:11, 864.71it/s]

    Metropolis sampling temp:  3


    100%|██████████| 10000/10000 [00:12<00:00, 805.89it/s]
      1%|          | 112/10000 [00:00<00:08, 1113.03it/s]

    Metropolis sampling temp:  6


    100%|██████████| 10000/10000 [00:12<00:00, 784.26it/s]
      1%|          | 94/10000 [00:00<00:10, 937.37it/s]

    Metropolis sampling temp:  9


    100%|██████████| 10000/10000 [00:13<00:00, 732.06it/s]
      0%|          | 37/10000 [00:00<00:34, 291.72it/s]

    Metropolis sampling temp:  12


    100%|██████████| 10000/10000 [00:11<00:00, 894.87it/s]
      1%|          | 91/10000 [00:00<00:10, 900.94it/s]

    Metropolis sampling temp:  15


    100%|██████████| 10000/10000 [00:13<00:00, 720.56it/s]
      1%|          | 93/10000 [00:00<00:10, 928.38it/s]

    Metropolis sampling temp:  18


    100%|██████████| 10000/10000 [00:10<00:00, 995.97it/s]
    <ipython-input-2-95628867df54>:86: RuntimeWarning: divide by zero encountered in double_scalars
      specific_heat = [np.var(samples[key]) / (float(key) ** 2) for key in samples.keys()]



    
![png](README_files/README_6_15.png)
    


You can see the specific heat decreases as a function of temperature, but total magnetism (sum of +1s and -1s on the lattice) stays constant over temperature. 


```python
num_rows = 20
num_cols = 20
num_iterations = 50000
J_over_kT = np.arange(0, 20, 3)
samples = {}

for ii_temp in J_over_kT:

    print('Metropolis sampling temp: ', ii_temp)
    initial_lattice = create_random_lattice(num_rows, num_cols)
    samples[str(ii_temp)] = metropolis_sampling(initial_lattice, num_iterations, ii_temp)

magnetism = compute_magnetism(samples)
plt.plot(J_over_kT, magnetism)
plt.xlabel('Temperature')
plt.ylabel('Magnetism')
plt.title('Magnetism over Temeprature')
plt.ylim((-(num_cols ** 2), num_cols ** 2))
plt.show()
```

      0%|          | 42/50000 [00:00<02:02, 409.32it/s]

    Metropolis sampling temp:  0


    100%|██████████| 50000/50000 [00:44<00:00, 1119.90it/s]
      0%|          | 116/50000 [00:00<00:43, 1156.87it/s]

    Metropolis sampling temp:  3


    100%|██████████| 50000/50000 [00:43<00:00, 1157.13it/s]
      0%|          | 121/50000 [00:00<00:41, 1206.19it/s]

    Metropolis sampling temp:  6


    100%|██████████| 50000/50000 [00:49<00:00, 1012.71it/s]
      0%|          | 119/50000 [00:00<00:41, 1188.70it/s]

    Metropolis sampling temp:  9


    100%|██████████| 50000/50000 [00:44<00:00, 1114.15it/s]
      0%|          | 122/50000 [00:00<00:41, 1214.50it/s]

    Metropolis sampling temp:  12


    100%|██████████| 50000/50000 [00:42<00:00, 1167.66it/s]
      0%|          | 115/50000 [00:00<00:43, 1145.08it/s]

    Metropolis sampling temp:  15


    100%|██████████| 50000/50000 [00:42<00:00, 1177.15it/s]
      0%|          | 125/50000 [00:00<00:40, 1241.70it/s]

    Metropolis sampling temp:  18


    100%|██████████| 50000/50000 [00:42<00:00, 1181.51it/s]



    
![png](README_files/README_8_15.png)
    

