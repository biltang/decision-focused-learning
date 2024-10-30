from torch import nn
import torch
from torch.autograd import Function
from torch.utils.data import Dataset
import numpy as np

from decision_learning.modeling.models import OptModel


# -------------------------------------------------------------------------
# SPO Plus (Smart Predict and Optimize Plus) Loss
# -------------------------------------------------------------------------
"""
Compared to PyEPO, try to minimize the coupling between solving the optimization problem and the dataset object,
and loss function as much as possible. Therefore, instead of having the optimization model solving for the true obj, sol
or each epoch predicted cost obj, sol within the dataset object or loss function, have the training loop handle that logic as much as possible.
Dataset object and loss function only need to deal with the input and output of the optimization model.
"""

class SPODataset(Dataset):
    """Torch Dataset for SPO Loss, which requires the following data:
        - pred_cost (torch.tensor): a batch of predicted values of the cost
        - opt_prob_sol (torch.tensor): a batch of optimal solutions with SPO specific objective
        - opt_prob_obj (torch.tensor): a batch of optimal objective values with SPO specific objective
        - true_cost (torch.tensor): a batch of true values of the cost
        - true_sol (torch.tensor): a batch of true optimal solutions
        - true_obj (torch.tensor): a batch of true optimal objective values      
    
    This dataset object will hold:
        - X - the features associated with each sample required for calculating pred_cost
        - true_cost
        - true_sol
        - true_obj
        
    opt_prob_sol and opt_prob_obj will be calculated by solving a optimization problem at each epoch using
    pred_cost, which is a function of X.
    
    Attributes:
        model (callable): a function/class that solves an optimization problem using pred_cost
        X (torch.tensor): a batch of features
        true_cost (torch.tensor): a batch of true values of the cost
        true_sol (torch.tensor): a batch of true optimal solutions
        true_obj (torch.tensor): a batch of true optimal objective values        
    """
    def __init__(self, X: np.ndarray, true_cost: np.ndarray, true_sol: np.ndarray, true_obj: np.ndarray):
        """Initialize the dataset

        Args:            
            X (np.ndarray): array of shape (n_samples, n_features) containing the features for each sample
            true_cost (np.ndarray): array of shape (n_samples, n_costs) containing the true cost vector for each sample
            true_sol (np.ndarray): array of shape (n_samples, n_costs) containing the true optimal solution for each sample
            true_obj (np.ndarray): array of shape (n_samples, 1) containing the true optimal objective value for each sample
        """
        # set data attributes        
        self.X = torch.tensor(X, dtype=torch.float32) #if not isinstance(X, torch.Tensor) else X
        self.true_cost = torch.tensor(true_cost, dtype=torch.float32) #if not isinstance(true_cost, torch.Tensor) else true_cost
        self.true_sol = torch.tensor(true_sol, dtype=torch.float32) #if not isinstance(true_sol, torch.Tensor) else true_sol
        self.true_obj = torch.tensor(true_obj, dtype=torch.float32) #if not isinstance(true_obj, torch.Tensor) else true_obj
            
    
    def __len__(self):
        return len(self.X)
    
    
    def __getitem__(self, idx):
        return self.X[idx], self.true_cost[idx], self.true_sol[idx], self.true_obj[idx]
    
    
    def collate_fn(self, batch):
        """Custom collate_fn so we can access data batches as named dictionaries to allow for better organization of data

        Args:
            batch: list of tensors returned by __getitem__

        Returns:
            _type_: _description_
        """
        X = torch.stack([item[0] for item in batch])
        true_cost = torch.stack([item[1] for item in batch])
        true_sol = torch.stack([item[2] for item in batch])
        true_obj = torch.stack([item[3] for item in batch])

        return {'X': X, 'true_cost': true_cost, 'true_sol': true_sol, 'true_obj': true_obj}

    
class SPOPlus(nn.Module):
    """
    Wrapper function around custom SPOLossFunc with customized forwards, backwards pass. Extend
    from nn.Module to use nn.Module's functionalities.
    
    An autograd module for SPO+ Loss, as a surrogate loss function of SPO Loss,
    which measures the decision error of the optimization problem.

    For SPO/SPO+ Loss, the objective function is linear and constraints are
    known and fixed, but the cost vector needs to be predicted from contextual
    data.

    The SPO+ Loss is convex with subgradient. Thus, it allows us to design an
    algorithm based on stochastic gradient descent.

    Reference: <https://doi.org/10.1287/mnsc.2020.3922>
    """

    def __init__(self, optmodel: callable, reduction: str = "mean", minimize: bool = True):
        """
        Args:
            optmodel (callable): a function/class that solves an optimization problem using pred_cost. For every batch of data, we use
                optmodel to solve the optimization problem using the predicted cost to get the optimal solution and objective value.
            reduction (str): the reduction to apply to the output
            minimize (bool): whether the optimization problem is minimization or maximization                    
        """
        super(SPOPlus, self).__init__()        
        self.spop = SPOPlusFunc()
        self.reduction = reduction
        self.minimize = minimize
        self.optmodel = optmodel

    def forward(self, 
            pred_cost: torch.tensor,             
            true_cost: torch.tensor, 
            true_sol: torch.tensor, 
            true_obj: torch.tensor):
        """
        Forward pass
        
        Args:            
            pred_cost (torch.tensor): a batch of predicted values of the cost            
            true_cost (torch.tensor): a batch of true values of the cost
            true_sol (torch.tensor): a batch of true optimal solutions
            true_obj (torch.tensor): a batch of true optimal objective values            
        """        
        loss = self.spop.apply(pred_cost, true_cost, true_sol, true_obj, self.optmodel, self.minimize)
        
        # reduction
        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        elif self.reduction == "none":
            loss = loss
        else:
            raise ValueError("No reduction '{}'.".format(self.reduction))
        return loss
    
    
class SPOPlusFunc(Function):
    """
    A autograd function for SPO+ Loss
    """

    @staticmethod
    def forward(ctx, 
            pred_cost: torch.tensor, 
            true_cost: torch.tensor, 
            true_sol: torch.tensor, 
            true_obj: torch.tensor,
            optmodel: callable,
            minimize: bool = True):
        """
        Forward pass for SPO+

        Args:
            ctx: Context object to store information for backward computation
            pred_cost (torch.tensor): a batch of predicted values of the cost            
            true_cost (torch.tensor): a batch of true values of the cost
            true_sol (torch.tensor): a batch of true optimal solutions
            true_obj (torch.tensor): a batch of true optimal objective values
            optmodel (callable): a function/class that solves an optimization problem using pred_cost. For every batch of data, we use
                optmodel to solve the optimization problem using the predicted cost to get the optimal solution and objective value.
            minimize (bool): whether the optimization problem is minimization or maximization

        Returns:
            torch.tensor: SPO+ loss
        """
        # rename variable names for convenience
        # c for cost, w for solution variables, z for obj values, and we use _hat for variables derived from predicted values
        c_hat = pred_cost
        c, w, z = true_cost, true_sol, true_obj
        
        # get batch's current optimal solution value and objective vvalue based on the predicted cost
        w_hat, z_hat = optmodel(2*c_hat - c)
                        
        # calculate loss
        # SPO loss = - min_{w} (2 * c_hat - c)^T w + 2 * c_hat^T w - z = - z_hat + 2 * c_hat^T w - z
        loss = - z_hat + 2 * torch.sum(c_hat * w, axis = 1).reshape(-1,1) - z
        
        # save solutions for backwards pass
        ctx.save_for_backward(w, w_hat)
        ctx.minimize = minimize
        
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for SPO+
        """
        w, w_hat = ctx.saved_tensors
  
        if ctx.minimize:
            grad = 2 * (w - w_hat)
        else:
            grad = 2 * (w_hat - w)
       
        return grad_output * grad, None, None, None, None, None, None

#------------------------------------------------    
# -Original - 
from pyepo.func.abcmodule import optModule
from pyepo import EPO

class SPOPlus2_O(optModule):
    """
    An autograd module for SPO+ Loss, as a surrogate loss function of SPO Loss,
    which measures the decision error of the optimization problem.

    For SPO/SPO+ Loss, the objective function is linear and constraints are
    known and fixed, but the cost vector needs to be predicted from contextual
    data.

    The SPO+ Loss is convex with subgradient. Thus, it allows us to design an
    algorithm based on stochastic gradient descent.

    Reference: <https://doi.org/10.1287/mnsc.2020.3922>
    """

    def __init__(self, optmodel, processes=1, solve_ratio=1, reduction="mean", dataset=None):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
            reduction (str): the reduction to apply to the output
            dataset (None/optDataset): the training data
        """
        super().__init__(optmodel, processes, solve_ratio, reduction, dataset)
        # build carterion
        self.spop = SPOPlusFunc_O()

    def forward(self, pred_cost, true_cost, true_sol, true_obj):
        """
        Forward pass
        """
        loss = self.spop.apply(pred_cost, true_cost, true_sol, true_obj, self)
        # reduction
        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        elif self.reduction == "none":
            loss = loss
        else:
            raise ValueError("No reduction '{}'.".format(self.reduction))
        return loss
class SPOPlusFunc_O(Function):
    """
    A autograd function for SPO+ Loss
    """

    @staticmethod
    def forward(ctx, pred_cost, true_cost, true_sol, true_obj, module):
        """
        Forward pass for SPO+

        Args:
            pred_cost (torch.tensor): a batch of predicted values of the cost
            true_cost (torch.tensor): a batch of true values of the cost
            true_sol (torch.tensor): a batch of true optimal solutions
            true_obj (torch.tensor): a batch of true optimal objective values
            module (optModule): SPOPlus modeul

        Returns:
            torch.tensor: SPO+ loss
        """
        # get device
        device = pred_cost.device
        # convert tenstor
        cp = pred_cost.detach().to("cpu")
        c = true_cost.detach().to("cpu")
        w = true_sol.detach().to("cpu")
        z = true_obj.detach().to("cpu")
        # check sol
        #_check_sol(c, w, z)
        # solve
        # sol, obj = _solve_or_cache(2 * cp - c, module)
        module.optmodel.setObj(2 * cp - c)
        sol, obj = module.optmodel.solve()
        # calculate loss
        loss = - obj + 2 * torch.sum(cp * w, axis = 1).reshape(-1,1) - z
        # sense
        if module.optmodel.modelSense == EPO.MAXIMIZE:
            loss = - loss
        # convert to tensor
        # save solutions
        ctx.save_for_backward(true_sol, sol)
        # add other objects to ctx
        ctx.optmodel = module.optmodel
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for SPO+
        """
        w, wq = ctx.saved_tensors
        optmodel = ctx.optmodel
        if optmodel.modelSense == EPO.MINIMIZE:
            grad = 2 * (w - wq)
        if optmodel.modelSense == EPO.MAXIMIZE:
            grad = 2 * (wq - w)
        return grad_output * grad, None, None, None, None