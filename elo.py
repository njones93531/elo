import numpy as np 
import math 
import random
import matplotlib.pyplot as plt 

#Probability elo1 beats elo2
def pwin(elo1, elo2, k):
    return 1 / (1 + math.pow(k, (elo2-elo1)/400))
#Find elo1' given outcome in ["win","draw","loss"]
def update(elo1, elo2, outcome, k):
    if outcome=="win":
        return int(elo1 + k *(1-pwin(elo1, elo2, k)))
    elif outcome=="draw":
        return int(elo1 + k *(0.5 - pwin(elo1, elo2, k)))
    else:
        return int(elo1 + k *(0 - pwin(elo1, elo2, k)))

def bounds(a):
    return max(1000,min(2800,a))

#Builds the transition matrix for a Markov chain of Elo scores
def construct_matrix(elo1, elo2, k):
    M = np.zeros((1801,1801), dtype=float)
    for f_elo in range(1000, 2801):
        M[bounds(f_elo)-1000,bounds(update(f_elo,elo2,"win", k))-1000]= pwin(f_elo, elo2, k)
        M[bounds(f_elo)-1000,bounds(update(f_elo,elo2,"lose", k))-1000]= 1-pwin(f_elo, elo2, k)  
    #print("Sum of M is:\n", sum(sum(M)))
    return M

#Checks if the diagonal entries of a matrix are nonzero
def check_full_diagonal(M):
    for i in range(0, len(M[0])):
        if M[i,i] == 0:
            return False
    return True

#Finds the period of a transition matrix of a markov chain, and returns
#both the period and the aperiodic power of the original matrix 
def find_period(M):
    i = 1
    B = M 
    while not check_full_diagonal(M):
        i+=1
        M = M @ B
        if i%1000==0:
            print("Testing matrix period:",i)
            print("Sum of M is:\n", np.sum(M, axis=1))
    return i, M

def weighted_mean(pi):
    return np.dot(pi, range(1000, 2801))


Elo_p = 1500.0
Elo_a_set = [1000, 1500, 2000]
k = 10
pi = []
pi_mean = []

for i, Elo_a in enumerate(Elo_a_set):
    M = construct_matrix(Elo_p, Elo_a, k)
    #print(M)
    period, Y = find_period(M) #Y = M^period
    A = Y
    print("Period is: ", period)
    A[:,-1] = np.ones(A[:,-1].shape) #Replace last column of A with ones
    A_inv = np.linalg.inv(A) #Find A^-1
    pi.append(A[-1,:]/sum(A[-1,:])) #Store the stationary distribution. Dividing by sum ensures pi is a probability distribution
    pi_mean.append(weighted_mean(pi[i]))
    print(f'For Elo_p = {Elo_p} and Elo_a = {Elo_a}, the stationary distribution is centered on {pi_mean[i]} for a difference of {pi_mean[i] - Elo_p}')


#Plot each pi
colors = ["r","g","b"]
labels = ["Elo_a: " + str(Elo_a) for Elo_a in Elo_a_set]
for i, y in enumerate(pi):
    print("Dist: ",y)
    plt.plot(range(1000,2801), y, color=colors[i], label=labels[i])
plt.legend()
plt.savefig("elo.png")
