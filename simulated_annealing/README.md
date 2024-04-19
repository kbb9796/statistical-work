# Simulated annealing

In the same way we can use Markov Chain Monte Carlo (MCMC) methods to sample analytically intractable posterior distributions, we can use simulated annealing to approximate combinatorially intractable solutions. For example, we use simulated annealing to approximate a solution to the famous traveling salesman problem (TSP). The TSP asks for the optimal path (shortest distance traveled) for a salesman given a map of n cities. Since there are n! possible paths, even for medium $n$, there are too many routes to enumerate and pick the shortest one. 

The algorithm works as follows. First, we initialize a random path to travel. At each subsequent iteration, we propose a new path by randomly transposing or shuffling cities. For our algorithm, we have two methods to propose new paths: (1) swap neighboring cities uniformly at random, or (2) select two cities uniformly at random and flip the order of the cities in between them. In our algorithm, we choose whatever random transposition of cities decreases the distance traveled the most. The algorithm accepts the proposed path according to the classic rules of the Metropolis-Hastings algorithm. 

It should be noted that, unlike the Metropolis sampler, the acceptance probability depends on an added temperature parameter. The temperature of the algorithm is high in early iterations, allowing the algorithm to explore less likely paths (i.e. paths that have higher distances to travel). As the algorithm iterates, the temperature cools down, allowing less moves to paths that increase the distance to travel. The temperature parameter and its cooling schedule let the simulated annealing algorithm explore the state space early on, but as the temperature cools, the algorithm makes less dramatic changes. We are able to find an approximate solution to TSP using several attempts of the algorithm. 

We code the algorithm below. 


```python
import numpy as np
import random
import math
import matplotlib.pyplot as plt


def swap(x): 
    
    n = len(x)
    i  = random.sample(range(0, n), 1)[0]
    if i == 0:
        j = 1
    elif i == n - 1:
        j = n- 2
    else:
        j = i + 1

    y = np.copy(x)
    y[[i, j]] = y[[j, i]]

    return y

def inverse_swap(x):
    
    n = len(x)
    y = np.copy(x)
    i = random.sample(range(0, n - 2), 1)[0]
    j = random.sample(range(i + 1, n), 1)[0]

    y[i:j] = np.flip(x[i:j], axis = 0)

    return y

def distance(x):
    distances = np.zeros(len(x))
    for i in range(len(x) - 1):
        distances[i] = np.linalg.norm(x[i+1] - x[i])
    distances[-1] = np.linalg.norm(x[0] - x[-1]) ## must return to starting position

    return np.sum(distances)

def acceptance_probability(E_new, E_old, Temp):
    if E_new <= E_old:
        prob = 1
    else:
        prob = np.exp(-(E_new - E_old) / Temp)
        
    return min(1, prob)

def New_Temp(Temp, a = .95):
    return Temp*a

def New_iterations(iterations, b = 1.05):
    x = iterations * b
    
    return math.ceil(x)

def min_configuration(max_stages = 30, iterations = 1, Temp = 10, x = None): # 75 works well for max stages, 5 iterations works 

    total_distance_traveled = np.array([])

    x_old = x 
    for i in range(max_stages):
        for j in range(iterations): 
            E_old = distance(x_old)
            x_new1 = swap(x_old)
            E_new1 = distance(x_new1)
            x_new2 = inverse_swap(x_old)
            E_new2 = distance(x_new2)
            if E_new1 <= E_new2:
                E_new = E_new1
                x_new = x_new1
            else:
                E_new = E_new2
                x_new = x_new2
            accept_prob = acceptance_probability(E_new = E_new, E_old = E_old, Temp = Temp)
            if np.random.uniform() <= accept_prob:
                x_old = x_new
            else:
                x_old = x_old
        Temp = New_Temp(Temp = Temp, a = .95)
        iterations = New_iterations(iterations = iterations, b = 1.05)
        distance_x = distance(x_old)
        total_distance_traveled = np.append(total_distance_traveled, distance_x)

        #if i % 10 == 0:

            #print('Temperature: ', Temp, 'Iteration: ', i, 'Distance to Travel: ', distance_x)

    return x_old, total_distance_traveled


def graph_config(x):

    x = np.concatenate((x, np.array([x[0]])), axis = 0)
    x_1, x_2 = x[:, 0], x[:, 1]
    plt.title("Final Configuration")
    plt.scatter(x_1, x_2)
    plt.plot(x_1, x_2)
    plt.show()
```


```python
# Generate a random configuration/ordering to start the algorithm

x = np.random.uniform(low = -20, high = 20, size = 200).reshape(-1, 2)
plt.plot(x[:, 0], x[:, 1])
plt.scatter(x[:, 0], x[:, 1])
plt.title('Initial Configuration')
plt.show()
```


    
![png](README_files/README_2_0.png)
    



```python
numTrials = 3
trials = []

for iiTrial in range(numTrials):

    tempMinConfig, tempDistanceToTravel = min_configuration(Temp = 10, max_stages=100, iterations = 1, x = x)
    tempDict = {'configuration': tempMinConfig, 'distance': tempDistanceToTravel}
    trials.append(tempDict)
```


```python
colors = ['red', 'blue', 'green']

for iiTrial in range(numTrials):

    plt.plot(trials[iiTrial]['distance'], color = colors[iiTrial], label = f"Trial {iiTrial:.1f}")
    
plt.title('Distance to Travel per Iteration')
plt.xlabel('Iteration')
plt.ylabel('Distance')
plt.legend()
plt.show()
```


    
![png](README_files/README_4_0.png)
    



```python
## Consider making gif for the updates 

for iiTrial in range(numTrials):

    plt.scatter(x[:, 0], x[:, 1])
    tempCoords = trials[iiTrial]['configuration']
    plt.plot(tempCoords[:, 0], tempCoords[:, 1], color = colors[iiTrial])
    plt.title(f'Best Configuration for Trial {iiTrial + 1}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
```


    
![png](README_files/README_5_0.png)
    



    
![png](README_files/README_5_1.png)
    



    
![png](README_files/README_5_2.png)
    

