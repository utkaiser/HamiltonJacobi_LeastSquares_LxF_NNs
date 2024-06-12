#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 14:38:54 2024

@author: carlosesteveyague
"""

import torch
import numpy as np


def unif_sample_sphere(N, dim = 2):
    
    X = torch.randn([N, dim])
    X_norms = X.norm(2, dim = -1).unsqueeze(-1)
    
    return X/X_norms

class data_gen_one_car_OCP:
    
    def __init__(self, R, r_target):
        
        
        self.r_target = r_target
        self.R = R
        
    def rand_int_points(self, d, nPDE):
        
        x = unif_sample_sphere(nPDE, dim = 2)
        
        r = self.r_target + d + (self.R - self.r_target - d)*torch.rand([nPDE, 1])
        
        x = r*x
        
        Theta = 2*torch.pi*torch.rand([nPDE, 1])
        
        return torch.cat([x, Theta], dim = -1)
    
    
    def rand_bound_points(self, nBoundary, percentage_target = .5):
        
        n_target = int(percentage_target*nBoundary)
        n_edges = nBoundary - n_target
        
        
        x_target = self.r_target*unif_sample_sphere(n_target)
        Theta_target = 2*torch.pi*torch.rand([n_target, 1])       
        X_target = torch.cat([x_target, Theta_target], dim = -1)

        
        x_out = self.R*unif_sample_sphere(n_edges)
        theta_out = 2*torch.pi*torch.rand([n_edges, 1])
        X_out = torch.cat([x_out, theta_out], dim = -1)
        
        return X_target, X_out


class data_gen_two_car_game:
    
    def __init__(self, P1_target, P2_target):
        
        self.P1_target = P1_target        
        self.P2_target = P2_target
        
    def rand_int_points(self, d, nPDE):
        
        x = unif_sample_sphere(nPDE)
        r = self.P1_target + d +(self.P2_target - self.P1_target - 2*d)*torch.rand([nPDE, 1])
        
        Theta = 2*torch.pi*torch.rand([nPDE, 2])
        
        return torch.cat([r*x, Theta], dim = -1)
    
    
    def rand_bound_points(self, nBoundary):
        
        x_p1 = self.P1_target*unif_sample_sphere(nBoundary)
        Theta_p1 = 2*torch.pi*torch.rand([nBoundary, 2])
        X_p1 = torch.cat([x_p1, Theta_p1], dim = -1)
        
        x_p2 = self.P2_target*unif_sample_sphere(nBoundary)
        Theta_p2 = 2*torch.pi*torch.rand([nBoundary, 2])
        X_p2 = torch.cat([x_p2, Theta_p2], dim = -1)
        
        return X_p1, X_p2