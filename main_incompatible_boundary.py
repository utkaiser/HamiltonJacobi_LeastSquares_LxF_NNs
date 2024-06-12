#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 20:20:30 2024

@author: carlosesteveyague
"""

import torch.optim as optim
import matplotlib.pyplot as plt
import torch

## Domain

side_length = 5.

from NeuralNetworks.NNs import FCFF_4L

dim = 2


NN = FCFF_4L([dim,40,40,40,40])



from PointSampling.Cube import data_gen_cube

domain = data_gen_cube(side_length, dim)


#%%

from Hamiltonians.Eikonal_LxF import Eikonal_sq_LF_multiD

"""
We solve the boundary value problem

|\grad u(x)| = f(x)    on \Omega
u(x) = g(x)               on \partial\Omega

the function f(x) has to be non-negative.
"""


half_side = side_length/2

def f(X):    
    return 1 #- 2.*torch.cos(4*X[:,0])

def g(X):
    out = torch.zeros(X.shape[0])
    
    out[X[:, 0] == -half_side] = 2.*(half_side - (X[X[:, 0] == -half_side][:,1]).abs())
    
    out[X[:, 0] == half_side] = 1.*(half_side - (X[X[:, 0] == half_side][:,1]).abs())
    
    return out

def g2(X):
    out = torch.zeros(X.shape[0])
    
    out[X[:, 0] == -half_side] = -half_side + X[X[:, 0] == -half_side][:,1]
    
    out[X[:, 0] == half_side] = half_side + X[X[:, 0] == half_side][:,1]
    
    out[X[:, 1] == -half_side] = -half_side + X[X[:, 1] == -half_side][:,0]
    
    out[X[:, 1] == half_side] = half_side + X[X[:, 1] == half_side][:,0]
    
    return out

    
training_params = {
    'numerical_scheme': Eikonal_sq_LF_multiD,
    
    'f': f,
    'g': g,
    'c': None,
    
    'delta': .7,
    'n_patch': 1, ## Set to 1 for dimension bigger than 4
    
    'alpha': 2.5, ## parameter for the viscosity
    'beta': 0.,  ## parameter for the +u_i term
    
    'optimizer': optim.SGD(NN.parameters(), lr = .02, momentum = .2),
    'num_iterations': 1000,
    'n_coloc_points': 100,
    'n_boundary_points': 50,
    'lambda': 2. #weight parameter for the boundary loss
    }


from Training.training import train


total_loss, PDE_loss, boundary_loss = train(NN, domain, training_params)


plt.plot(total_loss)
plt.title('Total loss')
plt.show()

plt.plot(PDE_loss)
plt.plot(boundary_loss)
plt.title('PDE + boundary loss')
plt.show()

#%%
## We retrain with smaller patch size and without viscosity


training_params['delta'] = .6
training_params['alpha'] = 2.1
training_params['beta'] = 0.

total_loss, PDE_loss, boundary_loss = train(NN, domain, training_params)

plt.plot(total_loss)
plt.title('Total loss')
plt.show()

plt.plot(PDE_loss)
plt.plot(boundary_loss)
plt.title('PDE + boundary loss')
plt.show()


#%%


from visualization.plots_cube import plot_slice, plot_2d_proj

n_grid = 100

X_axis = 0
Y_axis = 1

plot_2d_proj(X_axis, Y_axis, NN, n_grid, side_length)




#%%

# plot an optimal trajectory given an initial point

from visualization.feedback_traj_cube import optimal_traj_cube
from visualization.plots_cube import plot_traj_cube


#initial positions
x_0 = torch.tensor([[1., .1],
                    [-.5, -.6],
                    [0.75, -1.]])
t_step = 0.01

traj = []

for i in range(x_0.shape[0]):
    traj_i = optimal_traj_cube(x_0[i], NN, t_step, side_length)
    traj.append(traj_i)


plot_traj_cube(traj, X_axis, Y_axis, NN, n_grid, side_length)


