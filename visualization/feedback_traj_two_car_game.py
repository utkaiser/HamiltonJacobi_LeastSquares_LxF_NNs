#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 13:40:22 2024

@author: carlosesteveyague
"""
import torch

def ReedsShepp_two_car_game(x1, x2, NN, x_step, theta_step, side_length, r_target, rho, sigma, max_n_step = 1000):
        
    traj_p1 = x1.unsqueeze(0)
    traj_p2 = x2.unsqueeze(0)
        
    dist_to_edges = side_length/2 - max( x1[:-1].abs().max(), x2[:-1].abs().max() )
    
    dist_players = (x1[:-1] - x2[:-1]).norm()
    
    while (dist_to_edges > 0) and dist_players > r_target and traj_p1.shape[0] < max_n_step:
        
        sigma1, sigma2 = sigma
        rho1, rho2 = rho
        
        with torch.no_grad():
            X = torch.cat([x1[:2], x2[:2], x1[-1][None], x2[-1][None]])
            
            # get direction for theta1
            x1_up_theta = X.clone() 
            x1_up_theta[-2] += theta_step
            x1_down_theta = X.clone()
            x1_down_theta[-2] -= theta_step
            grad_theta_1 = (NN(x1_up_theta) - NN(x1_down_theta))/(2*theta_step)
            
            theta1_dir = -(1/rho1)*torch.sign(grad_theta_1)
            
            
            # get direction for x1
            x1_id = torch.zeros([2, 6])
            x1_id[:, :2] = torch.eye(2)    
            X1_up_x = X.unsqueeze(-2) + x_step*x1_id
            X1_down_x = X.unsqueeze(-2) - x_step*x1_id
                        
            grad_x1 = (NN(X1_up_x) - NN(X1_down_x))/(2*x_step)
            
            v1_theta = torch.tensor([torch.cos(X[-2]), torch.sin(X[-2])])
            x1_dir = -sigma1*torch.sign( torch.dot(grad_x1.squeeze(), v1_theta) )
            
            # get new position for P1
            v1 = torch.tensor([x_step*x1_dir*torch.cos(X[-2]), x_step*x1_dir*torch.sin(X[-2]), theta_step*theta1_dir])
            
            new_x1 = x1 + v1
            
            traj_p1 = torch.cat((traj_p1, new_x1.unsqueeze(0)), dim = 0)
            
            x1 = new_x1
            
            # get direction for theta1
            x2_up_theta = X.clone() 
            x2_up_theta[-1] = X[-1] + theta_step
            x2_down_theta = X.clone()
            x2_down_theta[-1] = X[-1] - theta_step
            grad_theta_2 = (NN(x2_up_theta) - NN(x2_down_theta))/(2*theta_step)
            
            theta2_dir = (1/rho2)*torch.sign(grad_theta_2)
            
            # get direction for x2
            x2_id = torch.zeros([2, 6])
            x2_id[:, 2:4] = torch.eye(2)    
            X2_up_x = X.unsqueeze(-2) + x_step*x2_id
            X2_down_x = X.unsqueeze(-2) - x_step*x2_id
            
            grad_x2 = (NN(X2_up_x) - NN(X2_down_x))/(2*x_step)
            
            v2_theta = torch.tensor([torch.cos(X[-1]), torch.sin(X[-1])])
            x2_dir =  sigma2*torch.sign( torch.dot(grad_x2.squeeze(), v2_theta) )
            
            # get new position for P2
            v2 = torch.tensor([x_step*x2_dir*torch.cos(X[-1]), x_step*x2_dir*torch.sin(X[-1]), theta_step*theta2_dir])
            
            new_x2 = x2 + v2
            
            traj_p2 = torch.cat((traj_p2, new_x2.unsqueeze(0)), dim = 0)
            
            x2 = new_x2
            
            dist_to_edges = side_length/2 - max( x1[:-1].abs().max(), x2[:-1].abs().max() )
    
            dist_players = (x1[:-1] - x2[:-1]).norm()
        
    return traj_p1, traj_p2



def ReedsShepp_two_car_game_4d(x1, x2, NN, x_step, theta_step, P1_target, P2_target, rho, sigma, max_n_step = 1000):
        
    traj_p1 = x1.unsqueeze(0)
    traj_p2 = x2.unsqueeze(0)
        
    sigma1, sigma2 = sigma
    rho1, rho2 = rho
    
    dist_players = (x1[:-1] - x2[:-1]).norm()
    
    while dist_players > P1_target and dist_players < P2_target and traj_p1.shape[0] < max_n_step:
        
        
        with torch.no_grad():
            X = torch.cat([x1[:2]-x2[:2], x1[-1][None], x2[-1][None]])
            
            # get grad w.r.t. theta1
            x1_up_theta = X.clone() 
            x1_up_theta[-2] += theta_step
            x1_down_theta = X.clone()
            x1_down_theta[-2] -= theta_step
            grad_theta_1 = (NN(x1_up_theta) - NN(x1_down_theta))/(2*theta_step)
            
            
            # get grad w.r.t. theta2
            x2_up_theta = X.clone() 
            x2_up_theta[-1] = X[-1] + theta_step
            x2_down_theta = X.clone()
            x2_down_theta[-1] = X[-1] - theta_step
            grad_theta_2 = (NN(x2_up_theta) - NN(x2_down_theta))/(2*theta_step)
            
            
            # get grad w.r.t. x
            x_id = torch.zeros([2, 4])
            x_id[:, :2] = torch.eye(2)    
            X_up_x = X.unsqueeze(-2) + x_step*x_id
            X_down_x = X.unsqueeze(-2) - x_step*x_id
                        
            grad_x = (NN(X_up_x) - NN(X_down_x))/(2*x_step)
            
            # get new position for P1
            v1_theta = torch.tensor([torch.cos(X[-2]), torch.sin(X[-2])])
            x1_dir = -sigma1*torch.sign( torch.dot(grad_x.squeeze(), v1_theta) )
            
            theta1_dir = -(1/rho1)*torch.sign(grad_theta_1)
            
            v1 = torch.tensor([x_step*x1_dir*torch.cos(X[-2]), x_step*x1_dir*torch.sin(X[-2]), theta_step*theta1_dir])
            
            new_x1 = x1 + v1
            
            traj_p1 = torch.cat((traj_p1, new_x1.unsqueeze(0)), dim = 0)
            
            x1 = new_x1
            
            
            # get new position for P2
            v2_theta = torch.tensor([torch.cos(X[-1]), torch.sin(X[-1])])
            x2_dir =  -sigma2*torch.sign( torch.dot(grad_x.squeeze(), v2_theta) )
            
            theta2_dir = (1/rho2)*torch.sign(grad_theta_2)
            
            v2 = torch.tensor([x_step*x2_dir*torch.cos(X[-1]), x_step*x2_dir*torch.sin(X[-1]), theta_step*theta2_dir])
            
            new_x2 = x2 + v2
            
            traj_p2 = torch.cat((traj_p2, new_x2.unsqueeze(0)), dim = 0)
            
            x2 = new_x2
    
            dist_players = (x1[:-1] - x2[:-1]).norm()
        
    return traj_p1, traj_p2


def Dubins_two_car_game_4d(x1, x2, NN, x_step, theta_step, P1_target, P2_target, rho, sigma, max_n_step = 1000):
        
    traj_p1 = x1.unsqueeze(0)
    traj_p2 = x2.unsqueeze(0)
        
    
    dist_players = (x1[:-1] - x2[:-1]).norm()
    
    while dist_players > P1_target and dist_players < P2_target and traj_p1.shape[0] < max_n_step:
        
        sigma1, sigma2 = sigma
        rho1, rho2 = rho
        
        with torch.no_grad():
            X = torch.cat([x1[:2]-x2[:2], x1[-1][None], x2[-1][None]])
            
            # get grad w.r.t. theta1
            x1_up_theta = X.clone() 
            x1_up_theta[-2] += theta_step
            x1_down_theta = X.clone()
            x1_down_theta[-2] -= theta_step
            grad_theta_1 = (NN(x1_up_theta) - NN(x1_down_theta))/(2*theta_step)
            
            
            # get grad w.r.t. theta2
            x2_up_theta = X.clone() 
            x2_up_theta[-1] = X[-1] + theta_step
            x2_down_theta = X.clone()
            x2_down_theta[-1] = X[-1] - theta_step
            grad_theta_2 = (NN(x2_up_theta) - NN(x2_down_theta))/(2*theta_step)
                        
            
            
            # get new position for P1
            #v1_theta = torch.tensor([torch.cos(X[-2]), torch.sin(X[-2])])
            x1_dir = sigma1 #*torch.sign( torch.dot(grad_x.squeeze(), v1_theta) )
            
            theta1_dir = -(1/rho1)*torch.sign(grad_theta_1)
            
            v1 = torch.tensor([x_step*x1_dir*torch.cos(X[-2]), x_step*x1_dir*torch.sin(X[-2]), theta_step*theta1_dir])
            
            new_x1 = x1 + v1
            
            traj_p1 = torch.cat((traj_p1, new_x1.unsqueeze(0)), dim = 0)
            
            x1 = new_x1
            
            
            # get new position for P2
            #v2_theta = torch.tensor([torch.cos(X[-1]), torch.sin(X[-1])])
            x2_dir =  sigma2 #*torch.sign( torch.dot(grad_x.squeeze(), v2_theta) )
            
            theta2_dir = (1/rho2)*torch.sign(grad_theta_2)
            
            v2 = torch.tensor([x_step*x2_dir*torch.cos(X[-1]), x_step*x2_dir*torch.sin(X[-1]), theta_step*theta2_dir])
            
            new_x2 = x2 + v2
            
            traj_p2 = torch.cat((traj_p2, new_x2.unsqueeze(0)), dim = 0)
            
            x2 = new_x2
    
            dist_players = (x1[:-1] - x2[:-1]).norm()
        
    return traj_p1, traj_p2