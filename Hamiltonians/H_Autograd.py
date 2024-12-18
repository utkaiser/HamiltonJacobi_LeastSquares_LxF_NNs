# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 14:50:35 2024

@author: usuario
"""

import torch
from torch.autograd.functional import hessian, jacobian


def grad_f(f, X):
    
    dim = X.shape[-1]
    
    if f(torch.zeros([1,dim])).requires_grad == True:
        Y = X.clone().detach().requires_grad_(True)
        return torch.autograd.grad(f(Y).sum(), Y, create_graph=True)[0]
    else:
        def fun(X):
            return f(X).sum()
        return jacobian(fun, X, create_graph=False)

def laplacian(Phi, X):
    
    def f(X):
        return Phi(X).sum()
    
    dim = X.shape[-1]
    n = torch.prod(torch.tensor(list(X.shape[:-1])))
    
    if Phi(torch.zeros([1,dim])).requires_grad == True:
        hess = hessian(f, X, create_graph=True)
    else:
        hess = hessian(f, X, create_graph=False)
    
    return hess.reshape(dim*n, dim*n).diag().reshape(X.shape).sum(-1)

def laplacian2(Phi, X):
    
    n = X.shape[0]
    
    out = torch.zeros(n)
    
    for i in range(n):
        
        out[i] = torch.diagonal(hessian(Phi, X[i], create_graph=True)).sum()
    
    return out


def Eikonal_sq_autograd(Phi, X, delta, alpha, beta = 0., c = None):
 
    gradU = grad_f(Phi, X)
    
    if c == None:
        gradU_norm = (gradU**2).sum(dim=-1)
    else:
        gradU_norm = c(X)*(gradU**2).sum(dim=-1)
    
    if alpha != 0:
        LapU = laplacian(Phi, X)
    else:
        LapU = 0.
       
    if beta == 0:
        return gradU_norm - alpha*(LapU)     
    else:
        U_center = Phi(X)
        return gradU_norm - alpha*(LapU) + beta*U_center.squeeze()