#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 09:47:36 2024

@author: carlosesteveyague
"""

import torch
import numpy as np

def unif_sample_sphere(N, dim):
    
    X = torch.randn([N, dim])
    X_norms = X.norm(2, dim = -1).unsqueeze(-1)
    
    return X/X_norms


class data_gen_ball:
    
    def __init__(self, radii, dim):
        
        self.radii = radii
        self.dim = dim
        
        r = np.array(radii)
        self.r_prob = r/r.sum()
        
        self.r_max = max(radii)
        self.r_min = min(radii)
    
    
    def rand_int_points(self, delta, nPDE):
        
        if self.r_min > 0:
            Rs = self.r_min + delta + (self.r_max - self.r_min - 2*delta)*torch.rand(nPDE)
        else:
            Rs = (self.r_max - delta)*torch.rand(nPDE)
        
        X = unif_sample_sphere(nPDE, self.dim)
        
        return X*Rs.unsqueeze(-1)
               
    def rand_bound_points(self, nBoundary):
        
        X = unif_sample_sphere(nBoundary, self.dim)
        
        r_s = np.random.choice(np.array(self.radii), nBoundary, p=self.r_prob)
        
        return torch.Tensor(r_s[:,None])*X


class data_gen_ball_unif:
    
    def __init__(self, radii, dim):
        
        self.radii = radii
        self.dim = dim
        
        r = np.array(radii)
        self.r_prob = r/r.sum()
        
        self.r_max = max(radii)
        self.r_min = min(radii)
        
        """
        Radial sampling for uniform distribution
        
        r = (a + b*y)**(1/d)
        
        where
        
        a = r_min**d
        b = r_max**d - r_min**d
        
        and y is uniformly distributed in [0,1]
        """
    
    
    def rand_int_points(self, delta, nPDE):
        
        
        if self.r_min > 0:
            #scaling parameters
            a = (self.r_min + delta)**self.dim
            b = (self.r_max - delta)**self.dim - a
            Rs = (a + b*torch.rand(nPDE))**(1/self.dim)
        else:
            b = (self.r_max - delta)**self.dim
            Rs = (b*torch.rand(nPDE))**(1/self.dim)
        
        X = unif_sample_sphere(nPDE, self.dim)
        
        return X*Rs.unsqueeze(-1)
            
        
        
            
    def rand_bound_points(self, nBoundary):
        
        X = unif_sample_sphere(nBoundary, self.dim)
        
        r_s = np.random.choice(np.array(self.radii), nBoundary, p=self.r_prob)
        
        return torch.Tensor(r_s[:,None])*X