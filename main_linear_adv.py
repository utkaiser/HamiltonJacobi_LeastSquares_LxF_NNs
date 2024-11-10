#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 17:07:22 2024

@author: carlosesteveyague
"""

import torch.optim as optim
import matplotlib.pyplot as plt
import torch
from time import time as t

from NeuralNetworks.NNs import FCFF_3L, FCFF_2L

from PointSampling.Cube import data_gen_cube
from visualization.plots_cube import plot_2d_proj, plot_level_set_cube #, plot_2d_proj_w
from visualization.cube_training import plot_2d_proj_w


from Hamiltonians.Eikonal_LxF import advection_LF_multiD

from Training.training import train
from error_test.cube_error import error_cube


dim = 2

side_length = 6.

domain = data_gen_cube(side_length, dim)

#Right-hand-side of the PDE
def f(X):    
    return 0.

#Boundary condition
def g(X):    
    return 0

delta_list = [.5, .2, .1, .1, .1]
alpha_list = [2., 2., 2., 2., 2.]
N_col_list = [80, 80, 80, 80, 80]
N_b_list = [20, 20, 20, 20, 20]
rounds = len(delta_list)

NN = FCFF_3L([dim,20,20])
#NN = FCFF_2L([dim,40])

training_params = {
    'numerical_scheme': advection_LF_multiD,

    'f': f,
    'g': g,
    'c': None,
    
    'beta': 0.,  ## parameter for the +u_i term
    
    'optimizer': optim.SGD(NN.parameters(), lr = .02, momentum = .2),
    'num_iterations': 500,
    'lambda': 2. #weight parameter for the boundary loss
    }





run_time_history = torch.zeros(rounds)

for i in range(rounds):
    
    training_params['alpha'] = alpha_list[i]
    training_params['delta'] = delta_list[i]
    training_params['n_coloc_points'] = N_col_list[i]
    training_params['n_boundary_points'] = N_b_list[i]
    
    t0 = t()
    total_loss, PDE_loss, boundary_loss = train(NN, domain, training_params)
    t1 = t() - t0 
 
    run_time_history[i] = t1
    
    plt.plot(total_loss)
    plt.plot(PDE_loss)
    plt.plot(boundary_loss)
    plt.show()
    
    X_axis = 0
    Y_axis = 1

    n_grid = 100
    plot_2d_proj(X_axis, Y_axis, NN, n_grid, side_length)
    plot_2d_proj_w(X_axis, Y_axis, NN, n_grid, side_length, training_params)


    

print('Run time:', run_time_history.sum())


