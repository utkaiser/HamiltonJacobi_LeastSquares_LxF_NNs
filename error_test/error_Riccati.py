# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 15:26:07 2024

@author: usuario
"""

import torch
from scipy.integrate import solve_ivp
import numpy as np

def error_Riccati(NN, P0, side_length, T, n_x, n_t):
    
    dim = P0.shape[0]
    
    P_t = Riccati_diff_Eq(T, P0, n_t, dim)
    
    t_list = torch.linspace(0,T, n_t)
        
    with torch.no_grad():
        
        x_data = -1 + 2*torch.rand([n_x, dim])
        x_data = .5*side_length*x_data
        
        X = torch.cat([ t_list.unsqueeze(-1).repeat_interleave(n_x, 0),
                        x_data.repeat(n_t, 1)], dim = -1).float()
        
        Y_hat = NN(X).squeeze().numpy()
        
        Y = np.zeros(n_x*n_t)
        
        PX = (x_data@P_t.transpose()).numpy().transpose()
        
        for i in range(n_t):
            
            V_t = (PX[i]*x_data.numpy()).sum(-1) - 1
            
            Y[i*n_x: (i+1)*n_x] = 0.5*V_t
        
        MSE = ((Y - Y_hat)**2).mean()
        
        L_inf_error = abs((Y - Y_hat)).max()
    
    return MSE, L_inf_error



def Riccati_diff_Eq(T, P0, n_t, dim):
    
    def Riccati(t, y):
        
        y = y.reshape(dim, dim)

        return (-np.matmul(y,y) - np.eye(dim)).reshape(-1)
    
    
    t_list = np.linspace(0,T, n_t)
    
    sol = solve_ivp(Riccati, [0,T], P0.reshape(-1), t_eval = t_list)

    return sol.y.transpose().reshape(-1, dim, dim)

