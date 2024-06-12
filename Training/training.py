#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 17:38:44 2023

@author: carlosesteveyague
"""

import torch
from tqdm import tqdm


def train(NN, domain, training_params):
        
    num_scheme = training_params['numerical_scheme']
    f = training_params['f'] ## Right hand side of the equation
    g = training_params['g'] ## Initial condition
    c = training_params['c'] ## Term multiplying |\nabla u(x)|    
    
    delta = training_params['delta']
    alpha = training_params['alpha']
    beta = training_params['beta']
    optimizer = training_params['optimizer']
    
    n_iter = training_params['num_iterations']
    
    nPDE = training_params['n_coloc_points']
    nBoundary = training_params['n_boundary_points']
    lam = training_params['lambda']    
    
    training_loss = torch.zeros(n_iter)
    training_PDE_loss = torch.zeros(n_iter)
    training_boundary_loss = torch.zeros(n_iter)
    
    for n in tqdm(range(n_iter)):
          
        interior_points = domain.rand_int_points(delta, nPDE)
        boundary_points = domain.rand_bound_points(nBoundary)
        
        optimizer.zero_grad()
        
        boundary_loss = ((NN(boundary_points).squeeze() - g(boundary_points))**2).mean()
        PDE_loss = ((num_scheme(NN, interior_points, delta, alpha, beta, c) - f(interior_points))**2).mean()
        
        loss =  PDE_loss + lam*boundary_loss
        loss.backward()
        optimizer.step()
        
        training_loss[n] = loss.detach()
        training_PDE_loss[n] = PDE_loss.detach()
        training_boundary_loss[n] = boundary_loss.detach()
    
    return training_loss, training_PDE_loss, training_boundary_loss



def train_time_dependent(NN, domain, training_params):
        
    num_scheme = training_params['numerical_scheme']
    f = training_params['f'] ## Right hand side of the equation
    g = training_params['g'] ## Initial condition
    c = training_params['c'] ## Term multiplying |\nabla u(x)|
    
    delta_x = training_params['delta x']
    delta_t = training_params['delta t']
    alpha = training_params['alpha']
    beta = training_params['beta']
    optimizer = training_params['optimizer']
    
    n_iter = training_params['num_iterations']
    
    nPDE = training_params['n_coloc_points']
    nBoundary = training_params['n_boundary_points']
    lam = training_params['lambda']    
    
    training_loss = torch.zeros(n_iter)
    training_PDE_loss = torch.zeros(n_iter)
    training_boundary_loss = torch.zeros(n_iter)
    
    for n in tqdm(range(n_iter)):
          
        interior_points = domain.rand_int_points(delta_x, delta_t, nPDE)
        boundary_points = domain.rand_bound_points(nBoundary)
        
        optimizer.zero_grad()
        
        boundary_loss = ((NN(boundary_points).squeeze() - g(boundary_points))**2).mean()
        PDE_loss = ((num_scheme(NN, interior_points, delta_x, delta_t, alpha, beta, c) - f(interior_points))**2).mean()
        
        loss =  PDE_loss + lam*boundary_loss
        loss.backward()
        optimizer.step()
        
        training_loss[n] = loss.detach()
        training_PDE_loss[n] = PDE_loss.detach()
        training_boundary_loss[n] = boundary_loss.detach()
    
    return training_loss, training_PDE_loss, training_boundary_loss

