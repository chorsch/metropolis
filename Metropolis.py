# Metropolis simulation for Ising model

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

#constant values
J = -1
T = 1
#k = 1.38 * 10 ** (-23)
k=1

def generate_matrix(size):
    spins = np.zeros((size, size))
    for i in range(0, size):
        for j in range(0, size):
            spins[i, j] = np.random.choice([-1, 1])
    return spins

def magnetization(spins, length, height):
    sum = 0;
    N = length * height;
    for i in range(0, length):
        for j in range(0, height):
            sum += spins[i,j]
    return sum/N

def deltaE(spins, i, j):
    max = spins[0].size-1
    spin = -1*spins[i,j]

    #Assumes boundaries loop over
    #special cases at i,j = 0 and i,j = max
    L = (j-1)%(max-1)
    R = (j+1)%(max-1)
    U = (i+1)%(max-1)
    D = (i-1)%(max-1)

    #Add the new spin interactions together to find
    #change in energy
    #the interactions will be with the
    #left, right, up, and down entries
    return 2*J*spin*(spins[i,L]+spins[i,R]+spins[U,j]+spins[D,j])


#picks a random entry and follows the procedure
#that decides whether or not it should switch signs
def swap_iteration(spins):
    #choose a random entry to consider
    i = np.random.randint(0,spins[0].size)
    j = np.random.randint(0, spins[0].size)

    #find change in energy if we were to swap it
    dE = deltaE(spins, i, j)

    #if energy decreases, swap it
    if( dE < 0 ):
        spins[i, j] *= -1

    else:
        # pick random num between 0 and 1
        r = np.random.rand()
        #Find the probability of swapping
        P = np.exp(-dE/(k*T))

        #if r is within that probability, swap
        if( r <= P ):
            spins[i, j] *= -1
            #print("swapped")
    return spins

#calculates the energy from all of the spin interactions
def get_energy(spins):
    e = 0
    max = spins[0].size-1
    #To find energy, add up the interaction energies
    for i in range(0, spins[0].size):
        for j in range(0, spins[0].size):
            e +=spins[i, j] * (spins[i, (j+1)%max] +                               spins[(i+1)%max, j])
    return J*e

def run(dims, iterations):
    #get a matrix filled with random 1's and -1's
    spins = generate_matrix(dims);

    plt.figure(1)
    plt.title("Initial spin distribution")
    sns.heatmap(spins)

    X = []
    E = []
    M = []
    avgE = []

    e_tot = 0

    for i in range(0, iterations):
        #add the iteration num to our x values for the plot
        X.append(i)
        #run the swap process on a random entry
        spins = swap_iteration(spins)
        #find the total energy
        e_tot = get_energy(spins)
        E.append(e_tot)
        #M.append(magnetization(spins, spins[0].size, spins[0].size))

    #Plot energy vs number of iterations
    plt.figure(3)
    plt.title("Energy vs iterations")
    plt.plot(X, E)
    #plt.show()
    plt.figure(2)
    plt.title("Spin dist. after performing all iterations")
    sns.heatmap(spins)


run(50, 10000)
