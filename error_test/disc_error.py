#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 20:08:38 2024

@author: carlosesteveyague
"""

import torch


def unif_sample_sphere(N, dim):
    
    X = torch.randn([N, dim])
    X_norms = X.norm(2, dim = -1).unsqueeze(-1)
    
    return X/X_norms


def error_ball(NN, R, n_grid):
    
    dim = NN.L1.in_features
    
    
    with torch.no_grad():
        
        r_grid = R*torch.linspace(0, 1, n_grid)**(1/dim) 
        
        X = unif_sample_sphere(n_grid, dim)
        
        X = r_grid.unsqueeze(-1)*X
        
        Y = R - r_grid
        Y_hat = NN(X).squeeze()
        
        MSE = ((Y - Y_hat)**2).mean()
        
        L_inf_error = (Y - Y_hat).abs().max()
    
    return MSE, L_inf_error


def error_annulus(NN, radii, n_grid):
    
    r_max = max(radii)
    r_min = min(radii)
    
    r_mid = (r_max + r_min)/2 
    
    dim = NN.L1.in_features
    
    
    with torch.no_grad():
        
        a = r_min**dim
        b = r_max**dim - a
        
        r_grid =  (a + b*torch.linspace(0, 1, n_grid))**(1/dim)
        
        
        X = unif_sample_sphere(n_grid, dim)
        
        X = r_grid.unsqueeze(-1)*X
        
        
        Y = r_grid.clone()
        
        Y[r_grid>r_mid] = r_max - r_grid[r_grid>r_mid]
        Y[r_grid<=r_mid] = r_grid[r_grid<=r_mid] - r_min
        
        Y_hat = NN(X).squeeze()
        
        MSE = ((Y - Y_hat)**2).mean()
        
        L_inf_error = (Y - Y_hat).abs().max()
    
    return MSE, L_inf_error
