#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 20:08:38 2024

@author: carlosesteveyague
"""

import torch
import matplotlib.pyplot as plt


def unif_sample_sphere(N, dim):
    
    X = torch.randn([N, dim])
    X_norms = X.norm(2, dim = -1).unsqueeze(-1)
    
    return X/X_norms


def error_ball(NN, R, n_points, display = False):
    
    dim = NN.L1.in_features
    
    
    with torch.no_grad():
        
        Rs = R*(torch.rand(n_points))**(1/dim)
        Rs[0] = 0.
        
        X = unif_sample_sphere(n_points, dim)
        
        X = Rs.unsqueeze(-1)*X
        
        Y = R - Rs
        Y_hat = NN(X).squeeze()
        
        MSE = ((Y - Y_hat)**2).mean()
        
        L_inf_error = (Y - Y_hat).abs().max()
        
        if display == True:
            
            with torch.no_grad():
                
                n = int(n_points**(1/2))
                r_grid = torch.linspace(0, R, n)
                theta_grid = torch.linspace(0, 2*torch.pi, n)
                Grid_r, Grid_theta = torch.meshgrid(r_grid, theta_grid)
                Grid = torch.zeros([n, n, dim])
                GridX = Grid_r * torch.cos(Grid_theta)
                GridY = Grid_r * torch.sin(Grid_theta)
                Grid[:,:,0] = GridX
                Grid[:,:,1] = GridY
                
                Y = R - Grid.norm(2, -1)
                Y_hat = Y_hat = NN(Grid).squeeze()
                error =  (Y - Y_hat).abs()

                plt.pcolormesh(GridX, GridY, error)
                plt.clim(0, 1)
                plt.colorbar()
                plt.title('Absolute error w.r.t. ground truth')
                plt.show()        
    
    return MSE, L_inf_error


def error_annulus(NN, radii, n_points):
        
    r_max = max(radii)
    r_min = min(radii)
    
    r_mid = (r_max + r_min)/2 
    
    dim = NN.L1.in_features
    
    
    with torch.no_grad():
        
        a = r_min**dim
        b = r_max**dim - a
        
        Rs =  (a + b*torch.rand(n_points))**(1/dim)
        
        
        X = unif_sample_sphere(n_points, dim)
        
        X = Rs.unsqueeze(-1)*X
        
        
        Y = Rs.clone()
        
        Y[Rs>r_mid] = r_max - Rs[Rs>r_mid]
        Y[Rs<=r_mid] = Rs[Rs<=r_mid] - r_min
        
        Y_hat = NN(X).squeeze()
        
        MSE = ((Y - Y_hat)**2).mean()
        
        L_inf_error = (Y - Y_hat).abs().max()
    
    return MSE, L_inf_error
