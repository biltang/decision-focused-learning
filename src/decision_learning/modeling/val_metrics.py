import torch 
import numpy as np
from decision_learning.utils import handle_solver


def decision_regret(pred_cost: torch.tensor, 
        true_cost: np.ndarray, 
        true_obj: np.ndarray,
        optmodel: callable,
        minimize: bool=True,        
        solver_kwargs: dict = {}):
    """To calculate the decision regret based on predicted coefficients/parameters for optimization model, we need following:
    1. predicted coefficients/parameters for optimization model    
    2. true coefficients/parameters for optimization model (needed to calculate objective value under the optimal solutions induced by predicted coefficients)
    3. true objective function value (what we are benchmarking against)
    4. optmodel - needed to calculate optimal solutions induced by predicted coefficients
    
    Args:
        pred_cost (torch.tensor): predicted coefficients/parameters for optimization model
        true_cost (torch.tensor): true coefficients/parameters for optimization model
        true_obj (torch.tensor): true objective function value
        optmodel (callable): a function/class that solves an optimization problem using pred_cost. For every batch of data, we use
                optmodel to solve the optimization problem using the predicted cost to get the optimal solution and objective value.
                It must take in:                                 
                    - pred_cost (torch.tensor): predicted coefficients/parameters for optimization model
                    - solver_kwargs (dict): a dictionary of additional arrays of data that the solver
                It must also:
                    - detach tensors if necessary
                    - loop or batch data solve 
                In practice, the user should wrap their own optmodel in the decision_learning.utils.handle_solver function so that
                these are all taken care of.        
        solver_kwargs (dict): a dictionary of additional arrays of data that the solver
    """    
    # get batch's current optimal solution value and objective vvalue based on the predicted cost
    w_hat, z_hat = optmodel(pred_cost, solver_kwargs=solver_kwargs) 
    
    # To ensure consistency, convert everything into a pytorch tensor
    w_hat = torch.as_tensor(w_hat, dtype=torch.float32)
    z_hat = torch.as_tensor(z_hat, dtype=torch.float32)
    true_cost = torch.as_tensor(true_cost, dtype=torch.float32)
    true_obj = torch.as_tensor(true_obj, dtype=torch.float32)
    
    # objective value of pred_cost induced solution (w_hat) based on true cost
    obj_hat = (w_hat * true_cost).sum(axis=1, keepdim=True)
        
    regret = (obj_hat - true_obj).sum()
    if not minimize:
        regret = -regret
    
    opt_obj_sum = torch.sum(torch.abs(true_obj)).item() + 1e-7
    return regret.item() / opt_obj_sum
        
    