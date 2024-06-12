#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 18:54:29 2024

@author: carlosesteveyague
"""

import torch

    

class data_gen_cube:
    
    def __init__(self, side_length, dim):
        
        self.side_length = side_length
        self.dim = dim
    
    def rand_int_points(self, d, nPDE):
        
        X = -1 + 2*torch.rand([nPDE, self.dim])
        
        return (self.side_length/2 - d)*X
    
    
    def rand_bound_points(self, nBoundary):
        
        X = -self.side_length/2 + self.side_length*torch.rand([nBoundary, self.dim])
        
        idx0 = torch.arange(nBoundary)
        idx1 = torch.randint(self.dim, [nBoundary])
        idx_sgn = torch.randint(2, [nBoundary])
        
        X[idx0[idx_sgn == 0], idx1[idx_sgn == 0]] = self.side_length/2
        
        X[idx0[idx_sgn == 1], idx1[idx_sgn == 1]] = -self.side_length/2
                
        return X