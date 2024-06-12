#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 11:37:07 2024

@author: carlosesteveyague
"""

import torch.optim as optim
import matplotlib.pyplot as plt
import torch
from time import time as t


from NeuralNetworks.NNs import periodic_3L_two_players


n_freq = 4


NN = periodic_3L_two_players([4,80,80], n_freq)

from PointSampling.Cars import data_gen_two_car_game


P1_target = .2
P2_target = 4.

domain = data_gen_two_car_game(P1_target, P2_target)




from Hamiltonians.Cars import LxF_ReedsShepp_two_car_game

def f(X):    
    return 1.

def g_p1(x):
    return 0.

def g_p2(x):
    return 10.

# maximum angular velocity for the cars
# [Pursuer, Evader]
rho = [.8, 1.]

# maximum velocity for the cars
sigma = [1., .8]

#%%
    
training_params = {
    'numerical_scheme': LxF_ReedsShepp_two_car_game,
    'rho': rho,
    'sigma': sigma,
    
    'f': f,
    'g P1': g_p1,
    'g P2': g_p2,
    
    'beta': 0.,
    
    'optimizer': optim.SGD(NN.parameters(), lr = .01, momentum = .2),
    'num_iterations': 3000,
    
    'lambda target P1': 1., #weight parameter for the boundary loss associated with the P1's target
    'lambda target P2': 1. #weight parameter for the boundary loss associated with the P2's target
    }


from Training.Cars_training import train_two_car_game

delta_list = [.7, .5, .3]
alpha_list = [2.5, 2., 1.5]
N_col_list = [800, 800, 800]
N_b_list = [100, 100, 100]
rounds = len(delta_list)

t1 = t()
for i in range(rounds):
    
    training_params['alpha'] = alpha_list[i]
    training_params['delta x'] = delta_list[i]
    training_params['delta theta'] = delta_list[i]
    training_params['n_coloc_points'] = N_col_list[i]
    training_params['n_boundary_points'] = N_b_list[i]



    total_loss, PDE_loss, boundary_loss = train_two_car_game(NN, domain, training_params)


    plt.plot(total_loss)
    plt.title('Total loss')
    plt.show()

    plt.plot(PDE_loss)
    plt.plot(boundary_loss)
    plt.title('PDE loss + Boundary loss')
    plt.show()

#torch.save(NN.state_dict(), 'trained_models/ReedShepp_loop')

#%%


from visualization.feedback_traj_two_car_game import ReedsShepp_two_car_game_4d

from visualization.plots_cube import plot_traj_game

#NN = periodic_3L_two_players([4,80,80], n_freq)
#NN.load_state_dict(torch.load('trained_models/ReedShepp_equal_rad'))
#NN.eval()

# Here we compute a feedback trajectory

x1 = torch.tensor([0., .5, 0.*torch.pi])
x2 = torch.tensor([0., -1., .5*torch.pi])


x_step = .05
theta_step = .05

trajectory_p1, trajectory_p2 = ReedsShepp_two_car_game_4d(x1, x2, NN, x_step, theta_step, P1_target, 
                                                       P2_target, rho, sigma, max_n_step = 1000)


side_length = 2*max(trajectory_p1[:, :-1].abs().max(), trajectory_p2[:, :-1].abs().max()) + 1.

plot_traj_game([trajectory_p1, trajectory_p2], side_length)

times = x_step*torch.arange(trajectory_p1.shape[0])

x1 = trajectory_p1[:, :2]
x2 = trajectory_p2[:, :2]
dists = (x1 - x2).norm(2,-1)
plt.plot(times, dists)
plt.show()


#%%
from visualization.plots_cube import plot_traj_game_movie



plot_traj_game_movie([trajectory_p1, trajectory_p2], side_length)