import torch 
def decision_regret(pred_cost: torch.tensor, 
        true_cost: torch.tensor, 
        true_obj: torch.tensor,
        optmodel: callable,):
    """To calculate the decision regret based on predicted coefficients/parameters for optimization model, we need following:
    1. predicted coefficients/parameters for optimization model    
    2. true coefficients/parameters for optimization model (needed to calculate objective value under the optimal solutions induced by predicted coefficients)
    3. true objective function value (what we are benchmarking against)
    4. optmodel - needed to calculate optimal solutions induced by predicted coefficients
    
    Args:
        pred_cost (torch.tensor): predicted coefficients/parameters for optimization model
        true_cost (torch.tensor): true coefficients/parameters for optimization model
        true_obj (torch.tensor): true objective function value
        optmodel (callable): optimization model
    """
    pass