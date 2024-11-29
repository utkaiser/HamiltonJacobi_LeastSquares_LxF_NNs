#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 16:06:49 2024

@author: carlosesteveyague
"""

import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def plot_2d_proj_time_dependent(axis1, axis2, NN, n_grid, t_grid, side_length, T):
    
    dim = NN.L1.in_features - 1
    
    Xgrid = torch.linspace(-side_length/2, side_length/2, n_grid)

    GridX, GridY = torch.meshgrid(Xgrid,Xgrid)
    
    GridT = torch.linspace(0, T, t_grid)
    
    Grid = torch.zeros([t_grid, n_grid, n_grid, dim + 1])
    
    Grid[:,:,:,1 + axis1] = GridX
    Grid[:,:,:,1 + axis2] = GridY
    Grid[:,:,:, 0] = GridT[:, None, None]
    
    with torch.no_grad():
        V = NN(Grid).squeeze()

        for i in range(GridT.shape[0]):
        
            plt.pcolormesh(GridX.detach(), GridY.detach(), V[i].detach())
            plt.title('Solution')
            plt.show()
            
            #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            #ax.plot_surface(np.array(GridX), np.array(GridY), np.array(V[i].detach()), cmap=cm.coolwarm, linewidth=0, antialiased=False)
            #ax.axes.set_zlim3d(bottom=-side_length/2, top=side_length/2)
            #plt.show()


def plot_level_set_time_dependent(axis1, axis2, NN, n_grid, t_grid, side_length, T, level = 0, P_t_Riccati = []):
    
    dim = NN.L1.in_features - 1
    
    Xgrid = torch.linspace(-side_length/2, side_length/2, n_grid)

    GridX, GridY = torch.meshgrid(Xgrid,Xgrid)
    
    GridT = torch.linspace(0, T, t_grid)
    
    Grid = torch.zeros([t_grid, n_grid, n_grid, dim + 1])
    
    Grid[:,:,:,1 + axis1] = GridX
    Grid[:,:,:,1 + axis2] = GridY
    Grid[:,:,:, 0] = GridT[:, None, None]
    
    if len( P_t_Riccati )>0:
        X = Grid[0, :,:, 1:].reshape(-1, dim).numpy()
        PX = (X@P_t_Riccati.transpose()).transpose()
    
    with torch.no_grad():
        V = NN(Grid).squeeze()

        fig, ax = plt.subplots()
        for i in range(GridT.shape[0]):
            
            if  len( P_t_Riccati )>0:
                t = GridT[i].numpy()
                V_t = (PX[i]*X).sum(-1).reshape(GridX.shape)
                CS = ax.contour(np.array(GridX), np.array(GridY),
                                V_t+t-1, [t], colors = ['green'], linestyles = 'dashed')
                #ax.clabel(CS, inline=True, fontsize=10)
            
            CS = ax.contour(np.array(GridX), np.array(GridY), 
                            np.array(V[i].detach()+GridT[i]), [GridT[i]], colors = ['red'])
            ax.clabel(CS, inline=True, fontsize=10)
        
        ax.set_aspect('equal', 'box')
        plt.show()
    
    return V