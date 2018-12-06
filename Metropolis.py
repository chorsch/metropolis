#!/usr/bin/env python

# Metropolis simulation for Ising model
import numpy as np
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import seaborn as sns; sns.set()

# constant values
# TODO: set up proper constants so that the dE is in meaningful units
J = -1 # energy unit
k = 1 # boltzmann factor

# colormap for seaborn
colors = ((0.0,0.0,0.0,1.0), (0.6,0.6,0.6,1.0)) # black and grey for spin up/down
cm = clr.LinearSegmentedColormap.from_list('Custom', colors, len(colors))

# returns matrix of spins, avg magnetization, initial interaction energy
def generate_matrix(l,w):
    spins = np.random.choice((-1,1), (l,w))
    kernel = np.array([[0,1,0], [1,0,1], [0,1,0]])
    energy = J*np.sum(convolve(spins, kernel, mode='wrap')*spins)
    magnetization = np.average(spins)
    return (spins, energy, magnetization)

def deltaE(spins, i, j, l, w):
    spin = -1*spins[i,j]

    # find positions of adjacent points, assume boundaries loop over
    L = (i-1)%(w)
    R = (i+1)%(w)
    U = (j+1)%(l)
    D = (j-1)%(l)

    # Add the new spin interactions with adj points to find change in energy
    return 2*J*spin*(spins[L,j] + spins[R,j] + spins[i,U] + spins[i,D])


# picks a random entry and follows the procedure to decide whether to  switch signs
def metropolis(spins, energy, magnetization, l, w, N, T, X, E, M, S):
    switches = 0
    for n in range(0,N):
        X.append(n)
        # choose a random entry to consider
        i = np.random.randint(0, w)
        j = np.random.randint(0, l)


        # find change in energy if we were to swap it
        dE = deltaE(spins, i, j, l, w)

        # if energy decreases, swap it
        if( dE < 0 ):
            spins[i, j] *= -1
            energy = energy + dE
            switches += 1
            magnetization += 2*spins[i,j]/(l*w)

        else:
            # pick random num between 0 and 1
            r = np.random.rand()
            # Find the probability of swapping
            P = np.exp(-dE/(k*T))

            # if r is within that probability, swap
            if( r <= P ):
                spins[i, j] *= -1
                energy = energy + dE
                switches += 1
                magnetization += 2*spins[i,j]/(l*w)
                #print("swapped")
        S.append(switches)
        M.append(magnetization)
        E.append(energy)
    return (X, E, M, S)

def run(dims, iterations):
    # Generate matrix, get initial conditions
    (spins, energy, mag) = generate_matrix(dims,dims);

    plt.figure(1)
    plt.title("Initial Spin Distribution")
    plt.axis('off')
    p0 = sns.heatmap(spins, cmap=cm)
    cbar= p0.collections[0].colorbar
    cbar.set_ticks([-1,1])
    cbar.set_ticklabels(["spin down", "spin up"])

    T = 1 # temperature

    X = [] # iteration number for energy vs iteration plot
    E = [] # total energy per iteration
    M = [] # mean magnetization
    S = [] # total number of swaps

    (X, E, M, S) = metropolis(spins, energy, mag, dims, dims, iterations, T, X, E, M, S) # swap process for random entry

    # Plot final characteristics
    p3 = plt.figure(3)
    plt.title("Change in Energy vs Iterations")
    plt.xlabel("iteration")
    plt.ylabel("dE")
    plt.plot(X, E)
    
    plt.figure(2)
    plt.title("Spin Distribution After " + str(iterations)  +  " Iterations")
    p2 = sns.heatmap(spins, cmap=cm)
    plt.axis('off')
    cbar= p2.collections[0].colorbar
    cbar.set_ticks([-1,1])
    cbar.set_ticklabels(["spin down", "spin up"])

    plt.figure(4)
    plt.title("Total Number of Swaps")
    plt.xlabel("iterations")
    plt.ylabel("swaps")
    plt.plot(X,S)

    plt.figure(5)
    plt.title("Mean Magnetization vs Iterations")
    plt.xlabel("iterations")
    plt.ylabel("mean magnetization")
    plt.plot(X,M)

    plt.show()

    return (E[-1], M[-1]) # tuple of energy, mean magnetization

run(50, 3000000)
