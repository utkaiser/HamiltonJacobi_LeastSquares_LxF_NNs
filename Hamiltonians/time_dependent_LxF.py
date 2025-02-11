#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 17:15:30 2024

@author: carlosesteveyague
"""

import torch


def Eikonal_LxF_Euler_explicit(Phi, X, delta_x, delta_t, alpha, beta = 0., c = None):
    
    """
    For a given function Phi, and a set of points X,
    computes the numerical Hamiltonian for

    u_t + H(x, grad(Phi(x)))   for x in X

    where
    H(x, grad(Phi(x))) = c(x)*|grad(Phi(x))| + beta*Phi(x)

    The numerical scheme is Lax-Friedrichs in x and Euler explicit in t.

    The parameters are:
     - delta_x, delta_t > 0 are the discretization parameter for x and t
     - alpha > 0
     - beta is a scalar
     - c: R^n ---> R^n is a function
    """
    
    dim = X.shape[-1] -1
    
    Id_x = torch.cat((torch.zeros([dim, 1]), torch.eye(dim)), dim = 1)
    
    X_up = X.unsqueeze(-2) + delta_x*Id_x
    X_down = X.unsqueeze(-2) - delta_x*Id_x
    
    U_center = Phi(X)
    U_up = Phi(X_up).squeeze()
    U_down = Phi(X_down).squeeze()
    
    if c == None:
        gradU_norm = ((U_up - U_down)/(2*delta_x)).norm(dim = -1)
    else:
        gradU_norm = c(X)*((U_up - U_down)/(2*delta_x)).norm(dim = -1)
    
    LapU = ((U_up + U_down - 2*U_center)/(2*delta_x)).sum(dim=-1)
    
    
    #Time-derivative
    Id_t = torch.zeros([1, dim + 1])
    Id_t[0,0] = 1
    X_t_up = X + delta_t*Id_t
    
    U_t_up = Phi(X_t_up).squeeze()
    
    dtU = (U_t_up - U_center.squeeze())/delta_t
    
    if beta == 0:
        return dtU + gradU_norm - alpha*(LapU)
    else:
        return dtU + gradU_norm - alpha*(LapU) + beta*U_center.squeeze()
        



def Eikonal_LxF_Euler_implicit(Phi, X, delta_x, delta_t, alpha, beta = 0., c = None):
    
    """
    For a given function Phi, and a set of points X,
    computes the numerical Hamiltonian for

    u_t + H(x, grad(Phi(x)))   for x in X

    where
    H(x, grad(Phi(x))) = c(x)*|grad(Phi(x))| + beta*Phi(x)

    The numerical scheme is Lax-Friedrichs in x and Euler implicit in t.

    The parameters are:
     - delta_x, delta_t > 0 are the discretization parameter for x and t
     - alpha > 0
     - beta is a scalar
     - c: R^n ---> R^n is a function
    """
    
    dim = X.shape[-1] -1
    
    Id_x = torch.cat((torch.zeros([dim, 1]), torch.eye(dim)), dim = 1)
    
    X_up = X.unsqueeze(-2) + delta_x*Id_x
    X_down = X.unsqueeze(-2) - delta_x*Id_x
    
    U_center = Phi(X)
    U_up = Phi(X_up).squeeze()
    U_down = Phi(X_down).squeeze()
    
    if c == None:
        gradU_norm = ((U_up - U_down)/(2*delta_x)).norm(dim = -1)
    else:
        gradU_norm = c(X)*((U_up - U_down)/(2*delta_x)).norm(dim = -1)
    
    LapU = ((U_up + U_down - 2*U_center)/(2*delta_x)).sum(dim=-1)
    
    
    #Time-derivative
    Id_t = torch.zeros([1, dim + 1])
    Id_t[0,0] = 1
    X_t_down = X - delta_t*Id_t
    
    U_t_down = Phi(X_t_down).squeeze()
    
    dtU = ( U_center.squeeze() - U_t_down )/delta_t
    
    if beta == 0:
        return dtU + gradU_norm - alpha*(LapU)
    else:
        return dtU + gradU_norm - alpha*(LapU) + beta*U_center.squeeze() 


def Eikonal_sq_LxF_Euler_explicit(Phi, X, delta_x, delta_t, alpha, beta = 0., c = None):
    
    """
    For a given function Phi, and a set of points X,
    computes the numerical Hamiltonian for

    u_t + H(x, grad(Phi(x)))   for x in X

    where
    H(x, grad(Phi(x))) = c(x)*|grad(Phi(x))|^2 + beta*Phi(x)

    The numerical scheme is Lax-Friedrichs in x and Euler explicit in t.

    The parameters are:
     - delta_x, delta_t > 0 are the discretization parameter for x and t
     - alpha > 0
     - beta is a scalar
     - c: R^n ---> R^n is a function
    """
    
    dim = X.shape[-1] -1
    
    Id_x = torch.cat((torch.zeros([dim, 1]), torch.eye(dim)), dim = 1)
    
    X_up = X.unsqueeze(-2) + delta_x*Id_x
    X_down = X.unsqueeze(-2) - delta_x*Id_x
    
    U_center = Phi(X, delta_t)
    U_up = Phi(X_up, delta_t).squeeze()
    U_down = Phi(X_down, delta_t).squeeze()
    
    if c == None:
        gradU_norm = (((U_up - U_down)/(2*delta_x))**2).sum(dim = -1)
    else:
        gradU_norm = c(X)*(((U_up - U_down)/(2*delta_x))**2).sum(dim = -1)
    
    LapU = ((U_up + U_down - 2*U_center)/(2*delta_x)).sum(dim=-1)
    
    
    #Time-derivative
    Id_t = torch.zeros([1, dim + 1])
    Id_t[0,0] = 1
    X_t_up = X + delta_t*Id_t
    
    U_t_up = Phi(X_t_up, delta_t).squeeze()
    
    dtU = (U_t_up - U_center.squeeze())/delta_t
    
    if beta == 0:
        return dtU + gradU_norm - alpha*(LapU)
    else:
        return dtU + gradU_norm - alpha*(LapU) + beta*U_center.squeeze()