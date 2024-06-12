#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 13:08:21 2024

@author: carlosesteveyague
"""

import torch
from torch.autograd.functional import jacobian
import sys

sys. setrecursionlimit(20000)

def optimal_traj_ball(x, NN, t_step, radii, traj = None, max_iter = 10000):
    
    R = max(radii)
    r = min(radii)    
    
    if traj == None:
        traj = x.unsqueeze(0)
        return optimal_traj_ball(x, NN, t_step, radii, traj)
    
    x_norm = x.norm(2)
    
    if x_norm < r or x_norm > R or traj.shape[0] > max_iter:
        return traj
    else:
        def Value(pos):
            return NN(pos)
        
        grad = jacobian(Value, x).squeeze()
        
        a = grad/grad.norm()
        
        new_x = x - t_step*a
        
        traj = torch.cat((traj, new_x.unsqueeze(0)), dim = 0)
        
        return optimal_traj_ball(new_x, NN, t_step, radii, traj)