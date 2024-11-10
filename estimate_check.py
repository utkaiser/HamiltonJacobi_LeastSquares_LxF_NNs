import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

from scipy.optimize import minimize

N_list = 10*(1 + np.arange(2)) + 3
alpha = 2

V_list = np.ones(N_list[-1])

sing_vals = np.zeros(N_list.shape[0])

estimate = np.zeros(N_list.shape[0])

for i in range(N_list.shape[0]):
    
    N = N_list[i]
    V = V_list[:N-1]
    
    A = np.diag(V, 1) - np.diag(V, -1)
    
    Lap = -2*np.diag(np.ones(N)) + np.diag(np.ones(N-1), 1) + np.diag(np.ones(N-1), -1)
    
    M = A + alpha*Lap
    
    U_mat, sing_val, V_mat = np.linalg.svd(M)
    
    
    sing_vals[i] = sing_val[-1]
    
    our_bound = 4*alpha*np.sin(np.pi/(2*N)) - 2
    estimate[i] = our_bound


plt.plot(N_list, sing_vals)
#plt.plot(N_list, estimate)

#%%

V = np.ones(9)

alpha = 2.

def sing_val(V):
    
    N = V.shape[0] + 1
    
    A = np.diag(V, 1) - np.diag(V, -1)
    
    Lap = -2*np.diag(np.ones(N)) + np.diag(np.ones(N-1), 1) + np.diag(np.ones(N-1), -1)
    
    M = A + alpha*Lap
    
    return np.linalg.svd(M)[1][-1]

a = minimize(sing_val, V)