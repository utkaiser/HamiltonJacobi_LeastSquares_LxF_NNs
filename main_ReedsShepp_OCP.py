#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 19:56:02 2024

@author: carlosesteveyague
"""

import torch.optim as optim
import matplotlib.pyplot as plt
import torch
from time import time as t


from NeuralNetworks.NNs import FCFF_3L_vec


n_freq = 10


NN = FCFF_3L_vec([3,40,40], n_freq)


from PointSampling.Cars import data_gen_one_car_OCP


side_length = 12.
r_target = .2

R = 5.

domain = data_gen_one_car_OCP(R, r_target)


# maximum angular velocity for the car
rho = 1.

#%%


from Hamiltonians.Cars import LxF_ReedsShepp_Car

def f(X):    
    return 1.

def g_int(x):

    return 0. 

def g_ext(x):

    return 4

    
training_params = {
    'numerical_scheme': LxF_ReedsShepp_Car,
    'rho': rho,
    
    'f': f,
    'g int': g_int,
    'g ext': g_ext,
    
    'beta': 0.,
    
    'optimizer': optim.SGD(NN.parameters(), lr = .001, momentum=.2),
    #'optimizer': optim.Adam(NN.parameters(), lr = .001),
    'num_iterations': 3000,
    'percentage of target points': .8,
    
    
    'lambda target': 2., #weight parameter for the boundary loss associated with the target
    'lambda edges': 1. #weight parameter for the boundary loss associated with the outer boundary
    }


from Training.Cars_training import train_Car_target


delta_list = [.75, .6, .3]
alpha_list = [2.5, 2.5, 2.5]
N_col_list = [800, 1000, 1200]
N_b_list = [100, 100, 100]
rounds = len(delta_list)




t1 = t()
for i in range(rounds):
    
    training_params['alpha'] = alpha_list[i]
    training_params['delta x'] = delta_list[i]
    training_params['delta theta'] = delta_list[i]
    training_params['n_coloc_points'] = N_col_list[i]
    training_params['n_boundary_points'] = N_b_list[i]
    
    total_loss, PDE_loss, boundary_loss = train_Car_target(NN, domain, training_params)

    plt.plot(total_loss)
    plt.title('Total loss')
    plt.show()

    
    plt.plot(boundary_loss)
    plt.plot(PDE_loss)
    plt.title('Boundary loss + PDE loss')
    plt.show()

print('Training time:', t() - t1)

#torch.save(NN.state_dict(), 'trained_models/ReedShepp_OCP')

#%%

from visualization.feedback_traj_cube import optimal_traj_ReedsShepp_car

from visualization.plots_cube import plot_traj_cube, plot_traj_car_circ


#NN = FCFF_3L_vec([3,60,60], n_freq)
#NN.load_state_dict(torch.load('trained_models/ReedShepp_OCP'))
#NN.eval()

# Here we compute a feedback trajectory

x0 = torch.tensor([3.5, 2., .5*torch.pi])

x_step = .05
theta_step = .05

r_target2 = r_target+.05

trajectory = optimal_traj_ReedsShepp_car(x0, NN, x_step, theta_step, R, 
                                        r_target2, rho, max_n_step = 1000)


n_grid = 100 

plot_traj_car_circ([trajectory], 0, 1, NN, n_grid, side_length, rho)



