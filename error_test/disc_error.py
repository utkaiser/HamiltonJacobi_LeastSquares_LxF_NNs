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


def error_ball(NN, R, n_points):
    
    dim = NN.L1.in_features
    
    
    with torch.no_grad():
        
        Rs = R*(torch.rand(n_points))**(1/dim)
        
        X = unif_sample_sphere(n_points, dim)
        
        X = Rs.unsqueeze(-1)*X
        
        Y = R - Rs
        Y_hat = NN(X).squeeze()
        
        MSE = ((Y - Y_hat)**2).mean()
        
        L_inf_error = (Y - Y_hat).abs().max()
    
    return MSE, L_inf_error


def error_annulus(NN, radii, n_points):
    
    ## To do
    
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
