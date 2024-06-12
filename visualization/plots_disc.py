#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 17:24:14 2024

@author: carlosesteveyague
"""

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import torch
import numpy as np



def plot_2d_proj_disc(axis1, axis2, NN, n_theta, n_r, radii, shift = 0.):
    
    dim = NN.L1.in_features
    
    if len(radii) == 2:
        r_grid = torch.linspace(min(radii), max(radii), n_r)
    elif len(radii) == 1:
            r_grid = torch.linspace(0, max(radii), n_r)
    
    
    
    theta_grid = torch.linspace(0, 2*torch.pi, n_theta)
    
    Grid_r, Grid_theta = torch.meshgrid(r_grid, theta_grid)

    Grid = torch.zeros([n_r, n_theta, dim]) + shift
    
    GridX = Grid_r * torch.cos(Grid_theta)
    GridY = Grid_r * torch.sin(Grid_theta)
    
    Grid[:,:,axis1] = GridX
    Grid[:,:,axis2] = GridY

    V = NN(Grid).squeeze()

    plt.pcolormesh(GridX.detach(), GridY.detach(), V.detach())
    plt.show()
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(np.array(GridX), np.array(GridY), np.array(V.detach()), cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.axes.set_zlim3d(bottom=0, top=max(radii))
    plt.show()


def plot_level_set_disc(axis1, axis2, NN, n_theta, n_r, radii, levels = [0], shift = 0.):
    
    dim = NN.L1.in_features
    
    if len(radii) == 2:
        r_grid = torch.linspace(min(radii), max(radii), n_r)
    elif len(radii) == 1:
            r_grid = torch.linspace(0, max(radii), n_r)
    
    
    
    theta_grid = torch.linspace(0, 2*torch.pi, n_theta)
    
    Grid_r, Grid_theta = torch.meshgrid(r_grid, theta_grid)

    Grid = torch.zeros([n_r, n_theta, dim]) + shift
    
    GridX = Grid_r * torch.cos(Grid_theta)
    GridY = Grid_r * torch.sin(Grid_theta)
    
    Grid[:,:,axis1] = GridX
    Grid[:,:,axis2] = GridY

    V = NN(Grid).squeeze()

    fig, ax = plt.subplots()   
    CS = ax.contour(np.array(GridX), np.array(GridY), np.array(V.detach()), levels)
    ax.clabel(CS, inline=True, fontsize=10)
    plt.show()


def plot_traj_disc(traj, axis1, axis2, NN, n_theta, n_r, radii, shift = 0.):
    
    dim = NN.L1.in_features
    
    if len(radii) == 2:
        r_grid = torch.linspace(min(radii), max(radii), n_r)
    elif len(radii) == 1:
            r_grid = torch.linspace(0, max(radii), n_r)
    
    
    
    theta_grid = torch.linspace(0, 2*torch.pi, n_theta)
    
    Grid_r, Grid_theta = torch.meshgrid(r_grid, theta_grid)

    Grid = torch.zeros([n_r, n_theta, dim]) + shift
    
    GridX = Grid_r * torch.cos(Grid_theta)
    GridY = Grid_r * torch.sin(Grid_theta)
    
    Grid[:,:,axis1] = GridX
    Grid[:,:,axis2] = GridY

    V = NN(Grid).squeeze()

    plt.pcolormesh(GridX.detach(), GridY.detach(), V.detach())
    
    
    for traj_i in traj:
        plt.scatter(traj_i[:,axis1], traj_i[:,axis2], c = 'red', s = 1.)
        plt.scatter(traj_i[0,axis1], traj_i[0,axis2], c = 'red', marker = '*', s = 100)
    plt.show()
    
