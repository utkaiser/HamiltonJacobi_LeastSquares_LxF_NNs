#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 15:40:07 2024

@author: carlosesteveyague
"""

import torch

def error_cube(NN, side_length, n_grid):
    
    dim = NN.L1.in_features
    
    if n_grid**dim > 10e6:
        # if the dimension is too high, we compute the error in a slice
        return error_cube_slice(NN, side_length, n_grid)
    
    with torch.no_grad():
        Xgrid = torch.linspace(-side_length/2, side_length/2, n_grid)
        
        Grid = torch.meshgrid([Xgrid]*dim)
        
        X = torch.cat([Grid[i][None] for i in range(len(Grid))], dim = 0).reshape([dim, -1]).transpose(0,1)
        
        Y = side_length/2 - X.abs().max(-1)[0]
        Y_hat = NN(X).squeeze()
        
        MSE = ((Y - Y_hat)**2).mean()
        
        L_inf_error = (Y - Y_hat).abs().max()
    
    return MSE, L_inf_error



def error_cube_slice(NN, side_length, n_grid, axis1 = 0, axis2 = 1):
    
    dim = NN.L1.in_features
    
    with torch.no_grad():
        Xgrid = torch.linspace(-side_length/2, side_length/2, n_grid)
        
        GridX, GridY = torch.meshgrid([Xgrid, Xgrid])
        Grid = torch.cat([GridX[:,:,None], GridY[:,:,None]], dim = -1).reshape([-1, 2])
        
        X = torch.zeros([Grid.shape[0], dim])
        
        X[:, axis1] = Grid[:,0]
        X[:, axis2] = Grid[:,1]
        
        Y = side_length/2 - X.abs().max(-1)[0]
        Y_hat = NN(X).squeeze()
        
        MSE = ((Y - Y_hat)**2).mean()
        
        L_inf_error = (Y - Y_hat).abs().max()
    
    return MSE, L_inf_error
    

