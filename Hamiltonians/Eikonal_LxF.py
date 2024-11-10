#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 15:02:04 2023

@author: carlosesteveyague
"""

import torch


def Eikonal_sq_LF_multiD(Phi, X, delta, alpha, beta = 0., c = None):
    
    """
    For a given function Phi, and a set of points X,
    computes the Lax-Friedrichs numerical Hamiltonian for 

    H(x, grad(Phi(x))) = c(x)*|grad(Phi(x))|^2 + beta*Phi(x),   for x in X

    The parameters are:
     - delta > 0 is the discretization parameter
     - alpha > 0
     - beta is a scalar
     - c: R^n ---> R^n is a function
    """
    
    dim = X.shape[-1]
    
    
    X_up = X.unsqueeze(-2) + delta*torch.eye(dim)
    X_down = X.unsqueeze(-2) - delta*torch.eye(dim)
    
    U_center = Phi(X)
    U_up = Phi(X_up).squeeze()
    U_down = Phi(X_down).squeeze()
    
    if c == None:
        gradU_norm = (((U_up - U_down)/(2*delta))**2).sum(dim=-1)
    else:
        gradU_norm = c(X)*(((U_up - U_down)/(2*delta))**2).sum(dim=-1)
        
    
    LapU = ((U_up + U_down - 2*U_center)/(2*delta)).sum(dim=-1)
       
    if beta == 0:
        return gradU_norm - alpha*(LapU)     
    else:
        return gradU_norm - alpha*(LapU) + beta*U_center.squeeze()
         

def advection_LF_multiD(Phi, X, delta, alpha, beta = 0., c = None):
    
    """
    For a given function Phi, and a set of points X,
    computes the Lax-Friedrichs numerical Hamiltonian for 

    H(x, grad(Phi(x))) = c(x)*|grad(Phi(x))|^2 + beta*Phi(x),   for x in X

    The parameters are:
     - delta > 0 is the discretization parameter
     - alpha > 0
     - beta is a scalar
     - c: R^n ---> R^n is a function
    """
    
    dim = X.shape[-1]
    
    
    X_up = X.unsqueeze(-2) + delta*torch.eye(dim)
    X_down = X.unsqueeze(-2) - delta*torch.eye(dim)
    
    U_center = Phi(X)
    U_up = Phi(X_up).squeeze()
    U_down = Phi(X_down).squeeze()
    
    if c == None:
        gradU_norm = ((U_up - U_down)/(2*delta)).sum(dim=-1)
    else:
        gradU_norm = (c(X)*((U_up - U_down)/(2*delta))).sum(dim=-1)
        
    
    LapU = ((U_up + U_down - 2*U_center)/(2*delta**2)).sum(dim=-1)
       
    if beta == 0:
        return gradU_norm - alpha*(LapU)     
    else:
        return gradU_norm - alpha*(LapU) + beta*U_center.squeeze()
