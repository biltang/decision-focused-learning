from torch import nn
import torch
from torch.autograd import Function

from decision_learning.modeling.models import OptModel


def lossfunc_optmodel_interace():
    pass

# 
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

    def __init__(self, optmodel: OptModel, reduction: str = "mean"):
        """
        Args:
            optmodel (OptModel): an optimization model that must implement functions for:
                - setObj: Set the optimization model's objective function.
                - solve: Solve the optimization model and return the optimal solution.             
        """
        super(SPOPlus, self).__init__()        
        self.spop = SPOPlusFunc()
        self.reduction = reduction

    def forward(self, pred_cost: torch.tensor, true_cost: torch.tensor, true_sol: torch.tensor, true_obj: torch.tensor):
        """
        Forward pass
        
        Args:            
            pred_cost (torch.tensor): a batch of predicted values of the cost
            true_cost (torch.tensor): a batch of true values of the cost
            true_sol (torch.tensor): a batch of true optimal solutions
            true_obj (torch.tensor): a batch of true optimal objective values            
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
            module: nn.Module):
        """
        Forward pass for SPO+

        Args:
            ctx: Context object to store information for backward computation
            pred_cost (torch.tensor): a batch of predicted values of the cost
            true_cost (torch.tensor): a batch of true values of the cost
            true_sol (torch.tensor): a batch of true optimal solutions
            true_obj (torch.tensor): a batch of true optimal objective values
            module (nn.Module): SPOPlus module that wraps this custom loss function

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
        
        
        module.optmodel.setObj(2 * cp - c)
        sol, obj = module.optmodel.solve()
        # calculate loss
        loss = - obj + 2 * torch.sum(cp * w, axis = 1).reshape(-1,1) - z
        # sense
        if module.optmodel.modelSense == EPO.MAXIMIZE:
            loss = - loss
        
        
        # convert to tensor
        loss = torch.FloatTensor(loss).to(device)
        sol = torch.FloatTensor(sol).to(device)
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