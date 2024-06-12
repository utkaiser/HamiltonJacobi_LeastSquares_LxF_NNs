#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 16:33:30 2024

@author: carlosesteveyague
"""

import torch
from torch.autograd.functional import jacobian

def optimal_traj_cube(x, NN, t_step, side_length, traj = None):
        
    if traj == None:
        traj = x.unsqueeze(0)
        return optimal_traj_cube(x, NN, t_step, side_length, traj)
    
    if x.abs().max() >= side_length/2:
        return traj
    else:
        def Value(pos):
            return NN(pos)
        
        grad = jacobian(Value, x).squeeze()
        
        a = grad/grad.norm()
        
        new_x = x - t_step*a
        
        traj = torch.cat((traj, new_x.unsqueeze(0)), dim = 0)
        
        return optimal_traj_cube(new_x, NN, t_step, side_length, traj)


def optimal_traj_Dubins_car(x, NN, x_step, theta_step, side_length, r_target, rho, max_n_step = 1000):
        
    
    traj = x.unsqueeze(0)

    
    while (x[:-1].abs().max() < side_length/2) and x[:-1].norm() > r_target and traj.shape[0] < max_n_step:
        
        
        with torch.no_grad():
            x_up_theta = x + theta_step*torch.tensor([0.,0., 1.])
            x_up_theta[-1] = x_up_theta[-1]%(2*torch.pi)
            x_down_theta = x - theta_step*torch.tensor([0.,0., 1.])
            x_down_theta[-1] = x_down_theta[-1]%(2*torch.pi)
            grad_theta = (NN(x_up_theta) - NN(x_down_theta))/(2*theta_step)
            
            theta_dir = -torch.sign(grad_theta)
            
            v = torch.tensor([x_step*torch.cos(x[-1]), x_step*torch.sin(x[-1]), theta_step*theta_dir/rho])
            
            new_x = x + v
            new_x[-1] = new_x[-1]%(2*torch.pi)
            
            traj = torch.cat((traj, new_x.unsqueeze(0)), dim = 0)
            
            x = new_x
        
    return traj



def optimal_traj_ReedsShepp_car(x, NN, x_step, theta_step, R, r_target, rho, max_n_step = 1000):
        
    traj = x.unsqueeze(0)
        
    
    while (x[:-1].norm() < R) and x[:-1].norm() > r_target and traj.shape[0] < max_n_step:
        
        with torch.no_grad():
            x_up_theta = x + theta_step*torch.tensor([0.,0., 1.])
            x_up_theta[-1] = x_up_theta[-1]%(2*torch.pi)
            x_down_theta = x - theta_step*torch.tensor([0.,0., 1.])
            x_down_theta[-1] = x_down_theta[-1]%(2*torch.pi)
            grad_theta = (NN(x_up_theta) - NN(x_down_theta))/(2*theta_step)
            
            theta_dir = -torch.sign(grad_theta)
            
            
            x_up_pos = x.unsqueeze(0) + x_step*torch.tensor([[1.,0., 0.], [0.,1.,0.]])
            x_down_pos = x.unsqueeze(0) - x_step*torch.tensor([[1.,0., 0.], [0.,1.,0.]])
            
            grad_pos = (NN(x_up_pos) - NN(x_down_pos))/(2*x_step)
            
            v_theta = torch.tensor([torch.cos(x[-1]), torch.sin(x[-1])])
            x_dir = - torch.sign( torch.dot(grad_pos.squeeze(), v_theta) )
            
            #v = torch.tensor([x_step*x_dir*torch.cos(x[-1]), x_step*x_dir*torch.sin(x[-1]), theta_step*theta_dir/rho])
            
            v_x = theta_dir*x_dir*rho*( torch.sin(x[-1] + theta_dir*x_step/rho) - torch.sin(x[-1]) )
            v_y = theta_dir*x_dir*rho*( torch.cos(x[-1]) - torch.cos(x[-1] + theta_dir*x_step/rho)  )
            v_theta = theta_step*theta_dir/rho
            
            v = torch.tensor([v_x, v_y, v_theta])
            
            new_x = x + v
            new_x[-1] = new_x[-1]%(2*torch.pi)
            
            traj = torch.cat((traj, new_x.unsqueeze(0)), dim = 0)
            
            x = new_x
        
    return traj



def optimal_traj_Dubins_car_obs(x, NN, x_step, theta_step, side_length, r_target, rho, max_n_step = 1000, obstacles = None):
        
    
    traj = x.unsqueeze(0)
    
    if obstacles != None:
        centers = obstacles[0]
        radii = obstacles[1]
        
        v_to_centers = x[:-1].unsqueeze(0) - centers
        dist = (v_to_centers.norm(2, -1) - radii).min()
        print(dist)
    else:
        dist = 1.

    
    while (x[:-1].abs().max() < side_length/2) and x[:-1].norm() > r_target and traj.shape[0] < max_n_step and dist > 0.:
        
        
        with torch.no_grad():
            x_up_theta = x + theta_step*torch.tensor([0.,0., 1.])
            x_up_theta[-1] = x_up_theta[-1]%(2*torch.pi)
            x_down_theta = x - theta_step*torch.tensor([0.,0., 1.])
            x_down_theta[-1] = x_down_theta[-1]%(2*torch.pi)
            grad_theta = (NN(x_up_theta) - NN(x_down_theta))/(2*theta_step)
            
            theta_dir = -torch.sign(grad_theta)
            
            v = torch.tensor([x_step*torch.cos(x[-1]), x_step*torch.sin(x[-1]), theta_step*theta_dir/rho])
            
            new_x = x + v
            new_x[-1] = new_x[-1]%(2*torch.pi)
            
            traj = torch.cat((traj, new_x.unsqueeze(0)), dim = 0)
            
            x = new_x
            
            if obstacles != None:
                centers = obstacles[0]
                radii = obstacles[1]
        
                v_to_centers = x[:-1].unsqueeze(0) - centers
                dist = (v_to_centers.norm(2, -1) - radii).min()
            else:
                dist = 1.
        
    return traj