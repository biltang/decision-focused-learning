import inspect
import os 
import time
from functools import wraps
import torch 

import logging 
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def handle_solver(pred_cost: torch.tensor,                   
                optmodel: callable, 
                solver_kwargs: dict={},                
                detach_tensor: bool=True,
                solver_batch_solve: bool=False):
    """Wrapper function to handle calling the optimization model solver. It will handle the following:
    1. Detach the tensors and convert them to numpy arrays if needed
    2. If solver_batch_solve is True, it will passs the entire batch of data to the optimization model solver 
    otherwise it will call the optimization model solver for each data point in the batch
    
    Handling solver_kwargs:
    We expect solver_kwargs to be a dictionary of additional arrays of data that the solver may need, 
    ex: {'coef_matrix': [B, N, M]}, where B is the batch size, N is the number of rows in the matrix, and M is the number of columns in the matrix.
        In this case, solver_kwargs['coef_matrix'][i] will be the coef_matrix for the i-th data point in the batch.                    
    In this case, we will filter out the invalid arguments for the optimization model solver, 
    and then depending on solver_batch_solve:
        - if solver_batch_solve is True, we will pass the entire batch of data to the optimization model solver
        - if solver_batch_solve is False, we will call the optimization model solver for each data point in the batch
            by extracting the i-th data point from solver_kwargs for each key in solver_kwargs
            
    Args:
        optmodel (callable): optimization model
        pred_cost (dict): predicted coefficients/parameters for optimization model
        solver_kwargs (dict): a dictionary of additional arrays of data that the solver
            may need to solve the optimization model. For example, solver_kwargs could look like:
            {'coef_matrix': [B, N, M]}, where B is the batch size, N is the number of rows in the matrix, and M is the number of columns in the matrix.
            In this case, solver_kwargs['coef_matrix'][i] will be the coef_matrix for the i-th data point in the batch.                    
        detach_tensor (bool): whether to detach the tensors and convert them to numpy arrays
        solver_batch_solve (bool): whether to pass the entire batch of data to the optimization model solver

    Returns:
        tuple: optimal solution value and objective value based on the predicted cost
    """
    if detach_tensor:
        pred_cost = pred_cost.detach().cpu().numpy()
        
        for key in solver_kwargs:
            if isinstance(solver_kwargs[key], torch.Tensor):
                solver_kwargs[key] = solver_kwargs[key].detach().cpu().numpy()

    # double check to ensure solver_kwargs only contains valid arguments for optmodel          
    solver_kwargs = filter_kwargs(optmodel, solver_kwargs)    
    if solver_batch_solve:                
        sol, obj = optmodel(pred_cost, **solver_kwargs)
        
    else:
        # if solver is not batch solve, we will call the solver for each data point in the batch
        sol = []
        obj = []
        for i in range(pred_cost.shape[0]): 
            # extract the i-th data point from solver_kwargs for each key in solver_kwargs
            cur_solver_kwargs = {k: v[i] for k, v in solver_kwargs.items()}
            
            sol_i, obj_i = optmodel(pred_cost[i], **cur_solver_kwargs)
            sol.append(sol_i)
            obj.append(obj_i)
        sol = np.array(sol)
        obj = np.array(obj)
    
    
    return sol, obj
    

def filter_kwargs(func: callable, kwargs: dict) -> dict:
    """Filter out the valid arguments for a function from a dictionary of arguments. This is useful when you want to
    pass a dictionary of arguments to a function, but only want to pass the valid arguments to the function. 

    Args:
        func (callable): function to filter arguments for
        kwargs (dict): dictionary of arguments to filter

    Returns:
        dict: dictionary of valid arguments for the function
    """
    signature = inspect.signature(func) # get the signature of the function
    valid_args = {key: value for key, value in kwargs.items() if key in signature.parameters} # filter out invalid args
    return valid_args


def log_runtime(func):
    """Decorator to log the runtime of a function"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Function '{func.__name__}' took {end_time - start_time} seconds to run.")
        return result
    return wrapper