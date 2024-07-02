#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 15:42:19 2024

@author: carlosesteveyague
"""

import torch


from NeuralNetworks.NNs import FCFF_3L_vec
from PointSampling.Cars import data_gen_one_car_OCP
from tqdm import tqdm


# number of sample trajectories
n_trajectories = int(1e+4)


side_length = 12.
r_target = .2

R = 5.

domain = data_gen_one_car_OCP(R, r_target)


# maximum angular velocity for the car
rho = 1.

from visualization.feedback_traj_cube import optimal_traj_ReedsShepp_car

from visualization.plots_cube import plot_traj_cube, plot_traj_car_circ


n_freq = 10

NN = FCFF_3L_vec([3,60,60], n_freq)
NN.load_state_dict(torch.load('trained_models/ReedShepp_OCP'))
NN.eval()

# Here we compute a feedback trajectory

x_step = .05
theta_step = .05
r_target2 = r_target+.05


initial_positions = domain.rand_int_points(2., n_trajectories)

trajectories = []

for i in tqdm(range(n_trajectories)):
    
    x0 = initial_positions[i]
    
    trajectory = optimal_traj_ReedsShepp_car(x0, NN, x_step, theta_step, R, 
                                        r_target2, rho, max_n_step = 200)
    if trajectory.shape[0] < 200:
        trajectories.append(trajectory)


n_grid = 100 
plot_traj_car_circ([trajectories[0]], 0, 1, NN, n_grid, side_length, rho)

