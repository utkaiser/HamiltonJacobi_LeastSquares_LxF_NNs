#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 10:26:40 2024

@author: carlosesteveyague
"""

import torch

def LxF_ReedsShepp_Car(Phi, X, delta_x, delta_theta, rho, alpha, beta):
    
    """
    This function computes the discretized Hamiltonian for the 
    smallest arrival tome for Reed-Shepp's Car
    
    The dimension is always 3:
        The first two dimensions represent the position, 
        and the third one the angle in [0,2pi)
    
    Input: 
        - Phi: function R^3 ---> R to be evaluated
        - X: torch tensor of shape [N, 3] with N collocation points
        - d: positive scalar, size of the patch for the local numerical scheme around each collocation point
        - alpha: positive scalar, coefficient of the upwind term in the Lax-Friedrichs numerical scheme
    
    Output:
        - torch tensor of shape [N] containing the discretized Hamltonian evaluated at the collocation points in N
    """
    
    
    dim = 2
    
    Theta = X[:, -1]
    
    x_id = torch.zeros([dim, dim + 1])
    x_id[:, :-1] = torch.eye(dim)
    
    X_up_x = X.unsqueeze(-2) + delta_x*x_id
    X_down_x = X.unsqueeze(-2) - delta_x*x_id
    
    U_center = Phi(X)
    U_up_x = Phi(X_up_x).squeeze()
    U_down_x = Phi(X_down_x).squeeze()
    
    gradU_x = (U_up_x - U_down_x)/(2*delta_x)
    
    LapU_x = ((U_up_x + U_down_x - 2*U_center)/(2*delta_x)).sum(dim=-1)
    
    
    theta_id = torch.zeros([1, dim + 1])
    theta_id[:, -1] = 1.
    
    X_up_theta = (X.unsqueeze(-2) + delta_theta*theta_id).squeeze()
    X_down_theta = (X.unsqueeze(-2) - delta_theta*theta_id).squeeze()
    
    # Modulus 2*pi
    X_up_theta[:, -1] = X_up_theta[:, -1]%(2*torch.pi)
    X_down_theta[:, -1] = X_down_theta[:, -1]%(2*torch.pi)
    
    
    U_up_theta = Phi(X_up_theta).squeeze()
    U_down_theta = Phi(X_down_theta).squeeze()
    
    gradU_theta = (U_up_theta - U_down_theta)/(2*delta_theta)
    
    LapU_theta = (U_up_theta + U_down_theta - 2*U_center.squeeze())/(2*delta_theta)
    
    
    CosSinTheta = torch.cat([torch.cos(Theta[:,None]), torch.sin(Theta[:,None])], dim = -1)
    
    x_term = (CosSinTheta*gradU_x).sum(-1).abs()
    theta_term = gradU_theta.abs()/rho
    
    Lap_term = LapU_x + LapU_theta
    
    if alpha>0:
        alpha_sgn = alpha/abs(alpha)
        return x_term + theta_term - alpha*Lap_term + alpha_sgn*beta*U_center.squeeze()
    else:
        return x_term + theta_term - alpha*Lap_term 


def LxF_Dubins_Car(Phi, X, delta_x, delta_theta, rho, alpha, beta):
    
    """
    This function computes the discretized Hamiltonian for the 
    smallest time arrivel for Reed-Shepp's Car
    
    The dimension is always 3:
        The first two dimensions represent the position, 
        and the third one the angle in [0,2pi)
    
    Input: 
        - Phi: function R^3 ---> R to be evaluated
        - X: torch tensor of shape [N, 3] with N collocation points
        - d: positive scalar, size of the patch for the local numerical scheme around each collocation point
        - alpha: positive scalar, coefficient of the upwind term in the Lax-Friedrichs numerical scheme
    
    Output:
        - torch tensor of shape [N] containing the discretized Hamltonian evaluated at the collocation points in N
    """
    
    
    dim = 2
    
    Theta = X[:, -1]
    
    x_id = torch.zeros([dim, dim + 1])
    x_id[:, :-1] = torch.eye(dim)
    
    X_up_x = X.unsqueeze(-2) + delta_x*x_id
    X_down_x = X.unsqueeze(-2) - delta_x*x_id
    
    U_center = Phi(X)
    U_up_x = Phi(X_up_x).squeeze()
    U_down_x = Phi(X_down_x).squeeze()
    
    gradU_x = (U_up_x - U_down_x)/(2*delta_x)
    
    LapU_x = ((U_up_x + U_down_x - 2*U_center)/(2*delta_x)).sum(dim=-1)
    
    
    theta_id = torch.zeros([1, dim + 1])
    theta_id[:, -1] = 1.
    
    X_up_theta = (X.unsqueeze(-2) + delta_theta*theta_id).squeeze()
    X_down_theta = (X.unsqueeze(-2) - delta_theta*theta_id).squeeze()
    
    # Modulus 2*pi
    X_up_theta[:, -1] = X_up_theta[:, -1]%(2*torch.pi)
    X_down_theta[:, -1] = X_down_theta[:, -1]%(2*torch.pi)
    
    
    U_up_theta = Phi(X_up_theta).squeeze()
    U_down_theta = Phi(X_down_theta).squeeze()
    
    gradU_theta = (U_up_theta - U_down_theta)/(2*delta_theta)
    
    LapU_theta = (U_up_theta + U_down_theta - 2*U_center.squeeze())/(2*delta_theta)
    
    
    CosSinTheta = torch.cat([torch.cos(Theta[:,None]), torch.sin(Theta[:,None])], dim = -1)
    
    x_term = -(CosSinTheta*gradU_x).sum(-1)  #.abs()
    theta_term = gradU_theta.abs()/rho
    
    Lap_term = LapU_x + LapU_theta
    
    if alpha>0:
        alpha_sgn = alpha/abs(alpha)
        return x_term + theta_term - alpha*Lap_term + alpha_sgn*beta*U_center.squeeze()
    else:
        return x_term + theta_term - alpha*Lap_term 



def LxF_ReedsShepp_two_car_game(Phi, X, delta_x, delta_theta, rho, sigma, alpha, beta):
    
    
    """
    
    First car is the pursuer and second car is the evader
    
    |cos(theta1) u_{x1} + sin(theta1) u_{y1}| + (1/rho) |u_{theta1}|
    - |cos(theta1) u_{x2} + sin(theta1) u_{y2}| - (1/rho) |u_{theta2}| - \Delta u = 1
    
    
    """
    
    rho1, rho2 = rho
    sigma1, sigma2 = sigma
    
    dim = 2
    
    # x derivatives
    x_id = torch.zeros([dim, dim + 2])
    x_id[:, :2] = torch.eye(dim)
    
    X_up_x = X.unsqueeze(-2) + delta_x*x_id
    X_down_x = X.unsqueeze(-2) - delta_x*x_id
    
    U_center = Phi(X)
    U_up_x = Phi(X_up_x).squeeze()
    U_down_x = Phi(X_down_x).squeeze()
    
    gradU_x = (U_up_x - U_down_x)/(2*delta_x)
    
    LapU_x = ((U_up_x + U_down_x - 2*U_center)/(2*delta_x)).sum(dim=-1)
    
    # theta derivatives for Player 1
    
    theta1_id = torch.zeros([1, dim + 2])
    theta1_id[:, -2] = 1.
    
    X1_up_theta = (X.unsqueeze(-2) + delta_theta*theta1_id).squeeze()
    X1_down_theta = (X.unsqueeze(-2) - delta_theta*theta1_id).squeeze()
    
    U_up_theta1 = Phi(X1_up_theta).squeeze()
    U_down_theta1 = Phi(X1_down_theta).squeeze()
    
    gradU_theta1 = (U_up_theta1 - U_down_theta1)/(2*delta_theta)
    
    LapU_theta1 = (U_up_theta1 + U_down_theta1 - 2*U_center.squeeze())/(2*delta_theta)
    
    # theta derivatives for Player 1
    
    theta2_id = torch.zeros([1, dim + 2])
    theta2_id[:, -1] = 1.
    
    X2_up_theta = (X.unsqueeze(-2) + delta_theta*theta2_id).squeeze()
    X2_down_theta = (X.unsqueeze(-2) - delta_theta*theta2_id).squeeze()
    
    U_up_theta2 = Phi(X2_up_theta).squeeze()
    U_down_theta2 = Phi(X2_down_theta).squeeze()
    
    gradU_theta2 = (U_up_theta2 - U_down_theta2)/(2*delta_theta)
    
    LapU_theta2 = (U_up_theta2 + U_down_theta2 - 2*U_center.squeeze())/(2*delta_theta)
    
    
    # H for Player 1
    Theta1 = X[:, -2]
    CosSinTheta1 = torch.cat([torch.cos(Theta1[:,None]), torch.sin(Theta1[:,None])], dim = -1)
    
    x1_term = sigma1*(CosSinTheta1*gradU_x).sum(-1).abs()
    theta1_term = gradU_theta1.abs()/rho1
    
    H1 =  x1_term + theta1_term
    
    # H for Player 2
    Theta2 = X[:, -1]
    CosSinTheta2 = torch.cat([torch.cos(Theta2[:,None]), torch.sin(Theta2[:,None])], dim = -1)
    
    x2_term = sigma2*(CosSinTheta2*gradU_x).sum(-1).abs()
    theta2_term = gradU_theta2.abs()/rho2
    
    H2 =  -x2_term - theta2_term
    
    Lap_term = LapU_x + LapU_theta1 + LapU_theta2
    
    if alpha>0:
        alpha_sgn = alpha/abs(alpha)
        return H1 + H2 - alpha*Lap_term + alpha_sgn*beta*U_center.squeeze()
    else:
        return H1 + H2 - alpha*Lap_term 
    
    
    
    
def LxF_Dubins_two_car_game(Phi, X, delta_x, delta_theta, rho, sigma, alpha, beta):
    
    
    """
    
    First car is the pursuer and second car is the evader
    
    -cos(theta1) u_{x1} - sin(theta1) u_{y1} + (1/rho) |u_{theta1}|
    + cos(theta2) u_{x2} + sin(theta2) u_{y2} - (1/rho) |u_{theta2}| - \Delta u = 1
    
    
    """
    
    rho1, rho2 = rho
    sigma1, sigma2 = sigma
    
    dim = 2
    
    # x derivatives
    x_id = torch.zeros([dim, dim + 2])
    x_id[:, :2] = torch.eye(dim)
    
    X_up_x = X.unsqueeze(-2) + delta_x*x_id
    X_down_x = X.unsqueeze(-2) - delta_x*x_id
    
    U_center = Phi(X)
    U_up_x = Phi(X_up_x).squeeze()
    U_down_x = Phi(X_down_x).squeeze()
    
    gradU_x = (U_up_x - U_down_x)/(2*delta_x)
    
    LapU_x = ((U_up_x + U_down_x - 2*U_center)/(2*delta_x)).sum(dim=-1)
    
    # theta derivatives for Player 1
    
    theta1_id = torch.zeros([1, dim + 2])
    theta1_id[:, -2] = 1.
    
    X1_up_theta = (X.unsqueeze(-2) + delta_theta*theta1_id).squeeze()
    X1_down_theta = (X.unsqueeze(-2) - delta_theta*theta1_id).squeeze()
    
    U_up_theta1 = Phi(X1_up_theta).squeeze()
    U_down_theta1 = Phi(X1_down_theta).squeeze()
    
    gradU_theta1 = (U_up_theta1 - U_down_theta1)/(2*delta_theta)
    
    LapU_theta1 = (U_up_theta1 + U_down_theta1 - 2*U_center.squeeze())/(2*delta_theta)
    
    # theta derivatives for Player 2
    
    theta2_id = torch.zeros([1, dim + 2])
    theta2_id[:, -1] = 1.
    
    X2_up_theta = (X.unsqueeze(-2) + delta_theta*theta2_id).squeeze()
    X2_down_theta = (X.unsqueeze(-2) - delta_theta*theta2_id).squeeze()
    
    U_up_theta2 = Phi(X2_up_theta).squeeze()
    U_down_theta2 = Phi(X2_down_theta).squeeze()
    
    gradU_theta2 = (U_up_theta2 - U_down_theta2)/(2*delta_theta)
    
    LapU_theta2 = (U_up_theta2 + U_down_theta2 - 2*U_center.squeeze())/(2*delta_theta)
    
    
    # H for Player 1
    Theta1 = X[:, -2]
    CosSinTheta1 = torch.cat([torch.cos(Theta1[:,None]), torch.sin(Theta1[:,None])], dim = -1)
    
    x1_term = sigma1*(CosSinTheta1*gradU_x).sum(-1)
    theta1_term = gradU_theta1.abs()/rho1
    
    H1 =  -x1_term + theta1_term
    
    # H for Player 2
    Theta2 = X[:, -1]
    CosSinTheta2 = torch.cat([torch.cos(Theta2[:,None]), torch.sin(Theta2[:,None])], dim = -1)
    
    x2_term = sigma2*(CosSinTheta2*gradU_x).sum(-1)
    theta2_term = gradU_theta2.abs()/rho2
    
    H2 =  x2_term - theta2_term
    
    Lap_term = LapU_x + LapU_theta1 + LapU_theta2
    
    if alpha>0:
        alpha_sgn = alpha/abs(alpha)
        return H1 + H2 - alpha*Lap_term + alpha_sgn*beta*U_center.squeeze()
    else:
        return H1 + H2 - alpha*Lap_term 
    
    
    
