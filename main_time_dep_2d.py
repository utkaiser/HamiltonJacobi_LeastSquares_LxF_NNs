#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 15:56:41 2024

@author: ce423
"""

import torch.optim as optim
import matplotlib.pyplot as plt
import torch
import numpy as np

from error_test.error_Riccati import Riccati_diff_Eq

## Domain

side_length = 6.
T = .5

from NeuralNetworks.NNs import FCFF_4L, FCFF_4L_modified_1, FCFF_4L_modified_2

dim = 2

NN_basic = FCFF_4L([dim+1,50,50,50])
NN_modified_1 = FCFF_4L_modified_1([dim+1,50,50,50])
NN_modified_2 = FCFF_4L_modified_2([dim+1,50,50,50])


from PointSampling.time_dependent import data_gen_cube_T

domain = data_gen_cube_T(side_length, T, dim)


#%%

from Hamiltonians.time_dependent_LxF import Eikonal_sq_LxF_Euler_explicit


"""
We solve the boundary value problem

u_t + c(x) |grad u|^2 = f(x)    on (0,T)x\R^d
u(x) =    g(x)            on \R^d

the function f(x) has to be non-negative.
"""


#Right-hand-side of the PDE
def f(X):    
    return -.5*(X**2).sum(-1) 


# Initial condition
a = torch.ones([1,dim])
a[:,0] = 4/25
a[:,1] = 1

def g(X):
    x = X[:, 1:]
    Ax = a*x 
    return .5*((x*Ax).sum(-1) - 1.)

def c(X):
    return .5 
    
training_params = {
    'numerical_scheme': Eikonal_sq_LxF_Euler_explicit,
    
    'f': f,
    'g': g,
    'c': c,
    
    'beta': 0.,  ## parameter for the +u term
    
    'optimizer': optim.SGD(NN_basic.parameters(), lr = .005, momentum = .2),
    
    'lambda': 1. #weight parameter for the boundary loss
    }


#%%
from Training.training import train_time_dependent
from visualization.plots_time_dependent import plot_level_set_time_dependent

delta_x_list = [.5, .3, .2, .1]
delta_t_list = [.05, .03, .02, .01]
alpha_list = [2.5, 2., 1.5, 1.]
N_col_list = [200, 200, 200, 200]
N_b_list = [40, 40, 40, 40]
num_iterations = [2000, 3000, 5000, 8000]
rounds = len(alpha_list)
n_grid = 100
t_grid = 6
X_axis = 0
Y_axis = 1

P0 = np.diag(a.squeeze())
P_t = Riccati_diff_Eq(T, P0, t_grid, dim)
total_loss_basic_mean, total_loss_modified_1_mean, total_loss_modified_2_mean = [], [], []

for i in range(rounds):
    print('------- epoch', i)
    training_params['delta x'] = delta_x_list[i]
    training_params['delta t'] = delta_t_list[i]
    training_params['alpha'] = alpha_list[i]
    training_params['n_coloc_points'] = N_col_list[i]
    training_params['n_boundary_points'] = N_b_list[i]
    training_params['num_iterations'] = num_iterations[i]
    training_params_modified_1 = training_params.copy()
    training_params_modified_2 = training_params.copy()

    training_params_modified_1['optimizer'] = optim.SGD(NN_modified_1.parameters(), lr = .005, momentum = .2)
    training_params_modified_2['optimizer'] = optim.SGD(NN_modified_2.parameters(), lr = .005, momentum = .2)

    total_loss_basic, PDE_loss_basic, boundary_loss_basic = train_time_dependent(NN_basic, domain, training_params)
    total_loss_modified_1, PDE_loss_modified_1, boundary_loss_modified_1 = train_time_dependent(NN_modified_1, domain, training_params_modified_1)
    total_loss_modified_2, PDE_loss_modified_2, boundary_loss_modified_2 = train_time_dependent(NN_modified_2, domain, training_params_modified_2)

    total_loss_basic_mean.append(total_loss_basic.mean())
    total_loss_modified_1_mean.append(total_loss_modified_1.mean())
    total_loss_modified_2_mean.append(total_loss_modified_2.mean())


    #plt.plot(total_loss)
    #plt.title('Total loss')
    #plt.show()
    
    #plt.plot(PDE_loss)
    #plt.plot(boundary_loss)
    #plt.title('PDE + boundary loss')
    #plt.show()

    #plot_level_set_time_dependent(X_axis, Y_axis, NN, n_grid, t_grid, side_length, T)
    #print('R(u) =', PDE_loss[-20:].mean())
    #print('L(u)=', boundary_loss[-20:].mean())
    
    plot_level_set_time_dependent(X_axis, Y_axis, {"NN_basic":NN_basic, "NN_modified_1":NN_modified_1, "NN_modified_2":NN_modified_2}, training_params['delta t'], n_grid, t_grid, side_length, T, P_t_Riccati = P_t) 

plt.plot(total_loss_basic_mean, marker = 'o', label = 'Basic')
plt.plot(total_loss_modified_1_mean, marker = 's', label = 'total_loss_modified_1')
plt.plot(total_loss_modified_2_mean, marker = 's', label = 'total_loss_modified_2')
plt.title('Total loss')
plt.legend()
plt.show()