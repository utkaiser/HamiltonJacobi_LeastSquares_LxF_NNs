#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 16:34:03 2024

@author: carlosesteveyague
"""

import torch
from tqdm import tqdm




def train_Car_target(NN, domain, training_params):
        
    num_scheme = training_params['numerical_scheme']
    f = training_params['f'] ## Right hand side of the equation
    g_int = training_params['g int']  
    g_ext = training_params['g ext']
    rho = training_params['rho']
    
    delta_x = training_params['delta x']
    delta_theta = training_params['delta theta']
    alpha = training_params['alpha']
    beta = training_params['beta']
    optimizer = training_params['optimizer']
    
    n_iter = training_params['num_iterations']
    
    nPDE = training_params['n_coloc_points']
    nBoundary = training_params['n_boundary_points']
    pctg_target = training_params['percentage of target points']
    
    lam_target = training_params['lambda target'] 
    lam_edges = training_params['lambda edges'] 
    
    training_loss = torch.zeros(n_iter)
    training_PDE_loss = torch.zeros(n_iter)
    training_boundary_loss = torch.zeros(n_iter)
    
    for n in tqdm(range(n_iter)):
          
        interior_points = domain.rand_int_points(delta_x, nPDE)
        X_target, X_out = domain.rand_bound_points(nBoundary, pctg_target)
        
        optimizer.zero_grad()
        
        boundary_loss_target = ((NN(X_target).squeeze() - g_int(X_target))**2).mean()
        boundary_loss_out = ((NN(X_out) - g_ext(X_out))**2).mean()
        
        PDE_loss = ((num_scheme(NN, interior_points, delta_x, delta_theta, rho, alpha, beta) 
                     - f(interior_points))**2).mean()
        
        loss =  PDE_loss + lam_target*boundary_loss_target + lam_edges*boundary_loss_out
        loss.backward()
        optimizer.step()
        
        training_loss[n] = loss.detach()
        training_PDE_loss[n] = PDE_loss.detach()
        training_boundary_loss[n] = boundary_loss_target.detach() + boundary_loss_out.detach()
    
    return training_loss, training_PDE_loss, training_boundary_loss


def train_two_car_game(NN, domain, training_params):
        
    num_scheme = training_params['numerical_scheme']
    f = training_params['f'] ## Right hand side of the equation
    g_p1 = training_params['g P1']
    g_p2 = training_params['g P2'] 
    rho = training_params['rho']
    sigma = training_params['sigma']
    
    delta_x = training_params['delta x']
    delta_theta = training_params['delta theta']
    alpha = training_params['alpha']
    beta = training_params['beta']
    optimizer = training_params['optimizer']
    
    n_iter = training_params['num_iterations']
    
    nPDE = training_params['n_coloc_points']
    nBoundary = training_params['n_boundary_points']
    
    lam_target_p1 = training_params['lambda target P1'] 
    lam_target_p2 = training_params['lambda target P2'] 
    
    training_loss = torch.zeros(n_iter)
    training_PDE_loss = torch.zeros(n_iter)
    training_boundary_loss = torch.zeros(n_iter)
    
    for n in tqdm(range(n_iter)):
          
        interior_points = domain.rand_int_points(delta_x, nPDE)
        X_p1, X_p2 = domain.rand_bound_points(nBoundary)
        
        optimizer.zero_grad()
        
        boundary_loss_target_p1 = ((NN(X_p1).squeeze() - g_p1(X_p1))**2).mean()
        boundary_loss_target_p2 = ((NN(X_p2).squeeze() - g_p2(X_p2))**2).mean()
        boundary_loss = lam_target_p1*boundary_loss_target_p1 + lam_target_p2*boundary_loss_target_p2
        
        PDE_loss = ((num_scheme(NN, interior_points, delta_x, delta_theta, rho, sigma, alpha, beta) 
                     - f(interior_points))**2).mean()
        
        loss =  PDE_loss + boundary_loss
        loss.backward()
        optimizer.step()
        
        training_loss[n] = loss.detach()
        training_PDE_loss[n] = PDE_loss.detach()
        training_boundary_loss[n] = boundary_loss.detach()
    
    return training_loss, training_PDE_loss, training_boundary_loss
