# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 14:50:35 2024

@author: usuario
"""

import torch
from torch.autograd.functional import hessian


def grad_f(f, X):
    
    Y = torch.tensor(X, requires_grad= True)
    
    return torch.autograd.grad(f(Y).sum(), Y, create_graph=True)

def laplacian(f, X):
    
    dim = X.shape[1]
    n = X.shape[0]
    
    hess = hessian(f, X, create_graph=True)
    
    return hess.reshape(dim*n, dim*n).diag().reshape([n, dim]).sum(-1)

def Eikonal_sq_autograd(Phi, X, delta, alpha, beta = 0., c = None):
 
    gradU = grad_f(Phi, X)[0] 
    
    if c == None:
        gradU_norm = (gradU**2).sum(dim=-1)
    else:
        gradU_norm = c(X)*(gradU**2).sum(dim=-1)
    
    def f(x):
        return Phi(X).sum()
    
    LapU = laplacian(f, X)
       
    if beta == 0:
        return gradU_norm - alpha*(LapU)     
    else:
        U_center = Phi(X)
        return gradU_norm - alpha*(LapU) + beta*U_center.squeeze()