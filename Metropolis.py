#!/usr/bin/env python

# Metropolis simulation for Ising model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import seaborn as sns; sns.set()

# constant values
# TODO: set up proper constants so that the dE is in meaningful units
J = -1 # energy unit
T = 1 # temp unit
k = 1 # boltzmann factor

# colormap for seaborn
colors = ((0.0,0.0,0.0,1.0), (0.6,0.6,0.6,1.0)) # black and grey for spin up/down
cm = clr.LinearSegmentedColormap.from_list('Custom', colors, len(colors))

S = []

# returns matrix of spins, avg magnetization, initial interaction energy
def generate_matrix(l,w):
    spins = np.zeros((w, l))
    e = 0
    magnetization = 0
    for i in range(0, w):
        for j in range(0, l):
            spins[i, j] = np.random.choice([-1, 1])
            magnetization += spins[i,j]
            e += spins[i, j] * (spins[i, (j+1)%l] + spins[(i+1)%w, j])
    return (spins, magnetization/(l*w), J*e)

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
def swap_iteration(spins, energy, l, w):
    # choose a random entry to consider
    i = np.random.randint(0, w)
    j = np.random.randint(0, l)

    global switches, S

    # find change in energy if we were to swap it
    dE = deltaE(spins, i, j, l, w)

    # if energy decreases, swap it
    if( dE < 0 ):
        spins[i, j] *= -1
        energy = energy + dE
        switches += 1

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
            #print("swapped")
    S.append(switches)
    return (spins, energy)

def run(dims, iterations):
    # Generate matrix, get initial conditions
    (spins, mag, energy) = generate_matrix(dims,dims);

    plt.figure(1)
    plt.title("Initial Spin Distribution")
    plt.axis('off')
    p0 = sns.heatmap(spins, cmap=cm)
    cbar= p0.collections[0].colorbar
    cbar.set_ticks([-1,1])
    cbar.set_ticklabels(["spin down", "spin up"])

    X = [] # iteration number for energy vs iteration plot
    E = [] # total energy per iteration
    M = []
    avgE = []

    for i in range(0, iterations):
        X.append(i)
        (spins, energy) = swap_iteration(spins, energy, dims, dims) # swap process for random entry
        E.append(energy)
        #M.append(magnetization(spins, spins[0].size, spins[0].size)) # TODO: add this as a piece of the swap process like energy

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

    plt.show()

switches = 0
run(100, 2000000)
