#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 21:57:43 2023

@author: carlosesteveyague
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FCFF_4L(nn.Module):
    def __init__(self, arch):
        super(FCFF_4L, self).__init__()
        
        self.L1 = nn.Linear(arch[0], arch[1], bias = True)
        self.L2 = nn.Linear(arch[1], arch[2], bias = True)
        self.L3 = nn.Linear(arch[2], arch[3], bias = True)
        self.L4 = nn.Linear(arch[3], 1, bias = True)
        
    def forward(self, X):
        
        Y = F.relu(self.L1(X))
        Y = F.relu(self.L2(Y))
        Y = F.relu(self.L3(Y))
        
        return self.L4(Y)


class FCFF_3L(nn.Module):
    def __init__(self, arch):
        super(FCFF_3L, self).__init__()
        
        self.L1 = nn.Linear(arch[0], arch[1], bias = True)
        self.L2 = nn.Linear(arch[1], arch[2], bias = True)
        self.L3 = nn.Linear(arch[2], 1, bias = True)
        
    def forward(self, X):
        
        Y = F.relu(self.L1(X))
        Y = F.relu(self.L2(Y))
        
        return self.L3(Y)
    
class FCFF_2L(nn.Module):
    def __init__(self, arch):
        super(FCFF_2L, self).__init__()
        
        self.L1 = nn.Linear(arch[0], arch[1], bias = True)
        self.L2 = nn.Linear(arch[1], 1, bias = True)
        
    def forward(self, X):
        
        Y = F.relu(self.L1(X))
         
        return self.L2(Y)



class ResNet15(nn.Module):
    def __init__(self, d, d_in, t_step):
        super(ResNet15, self).__init__()
        
        self.t_step = t_step
        
        self.L1 = nn.Linear(d, d_in, bias = True)
        self.L2 = nn.Linear(d_in, d_in, bias = True)
        self.L3 = nn.Linear(d_in, d_in, bias = True)
        self.L4 = nn.Linear(d_in, d_in, bias = True)
        self.L5 = nn.Linear(d_in, d_in, bias = True)
        self.L6 = nn.Linear(d_in, d_in, bias = True)
        self.L7 = nn.Linear(d_in, d_in, bias = True)
        self.L8 = nn.Linear(d_in, d_in, bias = True)
        self.L9 = nn.Linear(d_in, d_in, bias = True)
        self.L10 = nn.Linear(d_in, d_in, bias = True)
        self.L11 = nn.Linear(d_in, d_in, bias = True)
        self.L12 = nn.Linear(d_in, d_in, bias = True)
        self.L13 = nn.Linear(d_in, d_in, bias = True)
        self.L14 = nn.Linear(d_in, d_in, bias = True)
        self.L15 = nn.Linear(d_in, 1, bias = True)
        
        
    def forward(self, X):
        
        Y = F.relu(self.L1(X))
        Y = Y + self.t_step*F.relu(self.L2(Y))
        Y = Y + self.t_step*F.relu(self.L3(Y))
        Y = Y + self.t_step*F.relu(self.L4(Y))
        Y = Y + self.t_step*F.relu(self.L5(Y))
        Y = Y + self.t_step*F.relu(self.L6(Y))
        Y = Y + self.t_step*F.relu(self.L7(Y))
        Y = Y + self.t_step*F.relu(self.L8(Y))
        Y = Y + self.t_step*F.relu(self.L9(Y))
        Y = Y + self.t_step*F.relu(self.L10(Y))
        Y = Y + self.t_step*F.relu(self.L11(Y))
        Y = Y + self.t_step*F.relu(self.L12(Y))
        Y = Y + self.t_step*F.relu(self.L13(Y))
        Y = Y + self.t_step*F.relu(self.L14(Y))
        
        return self.L15(Y)


class FCFF_3L_vec(nn.Module):

    # This NN is used for the Cars' problem, which has to be periodic in theta

    def __init__(self, arch, n_freq):
        super(FCFF_3L_vec, self).__init__()
        
        self.L1 = nn.Linear(arch[0] - 1, arch[1], bias = True)
        self.L2 = nn.Linear(arch[1], arch[2], bias = True)
        self.L3 = nn.Linear(arch[2], 2*n_freq - 1, bias = True)
        
        self.freqs = torch.arange(n_freq)
        self.freqs_expand = torch.cat([self.freqs, self.freqs[1:]], dim = 0)
        
    def forward(self, X):
        
        X_reshaped = X.reshape([-1, 3])
        
        x = X_reshaped[:, :-1]
        theta = X_reshaped[:, -1]
        
        Y = F.relu(self.L1(x))
        Y = F.relu(self.L2(Y))
        Y = self.L3(Y)
        
        xi = theta[:, None]*self.freqs
        CosSin = torch.cat([torch.cos(xi), torch.sin(xi[:, 1:])], dim = -1)
        
        out = (Y*CosSin).sum(-1).reshape(X.shape[:-1])
        
        return out.unsqueeze(-1)
    
    def d_theta(self, X):
        
        X_reshaped = X.reshape([-1, 3])
        
        x = X_reshaped[:, :-1]
        theta = X_reshaped[:, -1]
        
        Y = F.relu(self.L1(x))
        Y = F.relu(self.L2(Y))
        Y = self.L3(Y)
        
        xi = theta[:, None]*self.freqs
        dCosSin = self.freqs_expand[None]*torch.cat([-torch.sin(xi), torch.cos(xi[:, 1:])], dim = -1)
        
        out = (Y*dCosSin).sum(-1).reshape(X.shape[:-1])
        
        return out.unsqueeze(-1)


class periodic_3L_two_players(nn.Module):

    # This NN is used for the two Car games, which has to be periodic in theta_e and theta_p

    def __init__(self, arch, n_freq):
        super(periodic_3L_two_players, self).__init__()
        
        self.L1 = nn.Linear(arch[0] - 2, arch[1], bias = True)
        self.L2 = nn.Linear(arch[1], arch[2], bias = True)
        self.L3 = nn.Linear(arch[2], (2*n_freq - 1)**2, bias = True)
        
        self.freqs = torch.arange(n_freq)
        self.freqs_expand = torch.cat([self.freqs, self.freqs[1:]], dim = 0)
        
    def forward(self, X):
        
        X_reshaped = X.reshape([-1, self.L1.weight.shape[1] + 2])
        n = X_reshaped.shape[0] 
        
        x = X_reshaped[:, :-2]
        theta1 = X_reshaped[:, -2]
        theta2 = X_reshaped[:, -1]
        
        Y = F.relu(self.L1(x))
        Y = F.relu(self.L2(Y))
        Y = self.L3(Y)
        
        xi1 = theta1[:, None]*self.freqs
        xi2 = theta2[:, None]*self.freqs
        CosSin1 = torch.cat([torch.cos(xi1), torch.sin(xi1[:, 1:])], dim = -1)
        CosSin2 = torch.cat([torch.cos(xi2), torch.sin(xi2[:, 1:])], dim = -1)
        CosSin = (CosSin1[:, None]*CosSin2[:,:,None]).reshape([n, -1])
        
        out = (Y*CosSin).sum(-1).reshape(X.shape[:-1])
        
        return out.unsqueeze(-1)