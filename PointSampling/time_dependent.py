#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 17:05:11 2024

@author: carlosesteveyague
"""

import torch

    

class data_gen_cube_T:
    
    def __init__(self, side_length, T, dim):
        
        self.side_length = side_length
        self.T = T
        self.dim = dim
    
    def rand_int_points(self, delta_x, delta_t, nPDE):
        
        X = -1 + 2*torch.rand([nPDE, self.dim])
        t = delta_t + (self.T - delta_t)*torch.rand([nPDE, 1])
        
        return torch.cat((t, (self.side_length/2 - delta_x)*X), dim = -1)
    
    
    def rand_bound_points(self, nBoundary):
        
        X = -1 + 2*torch.rand([nBoundary, self.dim])
        t = self.T*torch.zeros([nBoundary, 1])
        
        return torch.cat((t, (self.side_length/2)*X), dim = -1)