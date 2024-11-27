from torch import nn
import torch
from torch.autograd import Function
from torch.utils.data import Dataset
import numpy as np


# -------------------------------------------------------------------------
# SPO Plus (Smart Predict and Optimize Plus) Loss
# -------------------------------------------------------------------------

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
        # TODO: add support for detach and convert to numpy since optmodel may not be written for torch tensors
        # rename variable names for convenience
        # c for cost, w for solution variables, z for obj values, and we use _hat for variables derived from predicted values
        c_hat = pred_cost
        c, w, z = true_cost, true_sol, true_obj
        
        # get batch's current optimal solution value and objective vvalue based on the predicted cost
        w_hat, z_hat = optmodel(2*c_hat - c)
                        
        # calculate loss
        # SPO loss = - min_{w} (2 * c_hat - c)^T w + 2 * c_hat^T w - z = - z_hat + 2 * c_hat^T w - z
        loss = - z_hat + 2 * torch.sum(c_hat * w, axis = 1).reshape(-1,1) - z
        if not minimize:
            loss = - loss
        
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


# -------------------------------------------------------------------------
# Perturbation Gradient (PG) Loss
# -------------------------------------------------------------------------

class PG_Loss(nn.Module):
    """
    An autograd module for Perturbation Gradient (PG) Loss.

    Reference: <https://arxiv.org/pdf/2402.03256>
    """

    def __init__(self, 
                optmodel: callable, 
                h: float=1, 
                finite_diff_type: str='B', 
                reduction: str="mean", 
                minimize: bool=True):                 
        """
        Args:
            optmodel (callable): a function/class that solves an optimization problem using pred_cost. For every batch of data, we use
                optmodel to solve the optimization problem using the predicted cost to get the optimal solution and objective value.
            h (float): perturbation size/finite difference step size for zeroth order gradient approximation
            finite_diff_sch (str, optional): Specify type of finite-difference scheme:
                                            - Backward Differencing/PGB ('B')
                                            - Central Differencing/PGC ('C')
                                            - Forward Differencing/PGF ('F')
            reduction (str): the reduction to apply to the output
            minimize (bool): whether the optimization problem is minimization or maximization
        """
        # the finite difference step size h must be positive
        if h < 0:
            raise ValueError("h must be positive")
        # finite difference scheme must be one of the following
        if finite_diff_type not in ['B', 'C', 'F']:
            raise ValueError("finite_diff_type must be one of 'B', 'C', 'F'")
        
        super(PG_Loss, self).__init__()     
        self.pg = PGLossFunc()   
        self.h = h
        self.finite_diff_type = finite_diff_type
        self.reduction = reduction
        self.minimize = minimize
        self.optmodel = optmodel
        

    def forward(self, pred_cost: torch.tensor, true_cost: torch.tensor):
        """
        Forward pass
        
        Args:            
            pred_cost (torch.tensor): a batch of predicted values of the cost            
            true_cost (torch.tensor): a batch of true values of the cost
        """
        loss = self.pg.apply(pred_cost, 
                            true_cost, 
                            self.h, 
                            self.finite_diff_type, 
                            self.optmodel,
                            self.minimize)
        
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
    
    
class PGLossFunc(Function):
    """
    A autograd function for Perturbation Gradient (PG) Loss.
    """

    @staticmethod
    def forward(ctx, 
            pred_cost: torch.tensor, 
            true_cost: torch.tensor, 
            h: float, 
            finite_diff_type: str,
            optmodel: callable,
            minimize: bool = True):            
        """
        Forward pass for PG Loss

        Args:
            pred_cost (torch.tensor): a batch of predicted values of the cost
            true_cost (torch.tensor): a batch of true values of the cost
            h (float): perturbation size/finite difference step size for zeroth order gradient approximation
            finite_diff_sch (str, optional): Specify type of finite-difference scheme:
                                            - Backward Differencing/PGB ('B')
                                            - Central Differencing/PGC ('C')
                                            - Forward Differencing/PGF ('F')
            optmodel (callable): a function/class that solves an optimization problem using pred_cost. For every batch of data, we use
                optmodel to solve the optimization problem using the predicted cost to get the optimal solution and objective value.
            minimize (bool): whether the optimization problem is minimization or maximization
            
        Returns:
            torch.tensor: PG loss
        """        
        # detach (stops gradient tracking since we will compute custom gradient) and move to cpu. Do this since
        # generally the optmodel is probably cpu based
        cp = pred_cost.detach().to("cpu")
        c = true_cost.detach().to("cpu")

        # for PG loss with zeroth order gradients, we need to perturb the predicted costs and solve
        # two optimization problems to approximate the gradient, where there is a cost plus and minus perturbation
        # that changes depending on the finite difference scheme.
        if finite_diff_type == 'C': # central diff: (1/2h) * (optmodel(pred_cost + h*true_cost) - optmodel(pred_cost - h*true_cost))
            cp_plus = cp + h * c
            cp_minus = cp - h * c
            step_size = 1 / (2 * h)
        elif finite_diff_type == 'B': # back diff: (1/h) * (optmodel(pred_cost) - optmodel(pred_cost - h*true_cost))
            cp_plus = cp
            cp_minus = cp - h * c
            step_size = 1 / h
        elif finite_diff_type == 'F': # forward diff: (1/h) * (optmodel(pred_cost + h*true_cost) - optmodel(pred_cost))
            cp_plus = cp + h * c
            cp_minus = cp
            step_size = 1 / h

        # solve optimization problems
        sol_plus, obj_plus = optmodel(cp_plus) # Plus Perturbation Optimization Problem
        sol_minus, obj_minus = optmodel(cp_minus) # Minus Perturbation Optimization Problem
        
        # calculate loss
        loss = (obj_plus - obj_minus) * step_size
        if not minimize:
            loss = - loss
                
        # save solutions and objects needed for backwards pass to compute gradients
        ctx.save_for_backward(sol_plus, sol_minus)        
        ctx.minimize = minimize
        ctx.step_size = step_size
        return loss


    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for PG Loss
        """
        sol_plus, sol_minus = ctx.saved_tensors  
        step_size = ctx.step_size

        # below, need to move (sol_plus - sol_minus) to the same device as grad_output since sol_plus and sol_minus
        # are on cpu and it is possible that grad_output is on a different device
        grad = step_size * (sol_plus - sol_minus).to(grad_output.device)
        if not ctx.minimize: # maximization problem case
            grad = - grad
        
        return grad_output * grad, None, None, None, None, None, None
    
    
# -------------------------------------------------------------------------
# perturbed Fenchel-Young (FYL) Loss
# -------------------------------------------------------------------------
# TODO - refactor below loss
    
# class perturbedFenchelYoung(optModule):
#     """
#     An autograd module for Fenchel-Young loss using perturbation techniques. The
#     use of the loss improves the algorithmic by the specific expression of the
#     gradients of the loss.

#     For the perturbed optimizer, the cost vector need to be predicted from
#     contextual data and are perturbed with Gaussian noise.

#     The Fenchel-Young loss allows to directly optimize a loss between the features
#     and solutions with less computation. Thus, allows us to design an algorithm
#     based on stochastic gradient descent.

#     Reference: <https://papers.nips.cc/paper/2020/hash/6bb56208f672af0dd65451f869fedfd9-Abstract.html>
#     """

#     def __init__(self, optmodel, n_samples=10, sigma=1.0, processes=1,
#                  seed=135, solve_ratio=1, reduction="mean", dataset=None):
#         """
#         Args:
#             optmodel (optModel): an PyEPO optimization model
#             n_samples (int): number of Monte-Carlo samples
#             sigma (float): the amplitude of the perturbation
#             processes (int): number of processors, 1 for single-core, 0 for all of cores
#             seed (int): random state seed
#             solve_ratio (float): the ratio of new solutions computed during training
#             reduction (str): the reduction to apply to the output
#             dataset (None/optDataset): the training data
#         """
#         super().__init__(optmodel, processes, solve_ratio, reduction, dataset)
#         # number of samples
#         self.n_samples = n_samples
#         # perturbation amplitude
#         self.sigma = sigma
#         # random state
#         self.rnd = np.random.RandomState(seed)
#         # build optimizer
#         self.pfy = perturbedFenchelYoungFunc()

#     def forward(self, pred_cost, true_sol):
#         """
#         Forward pass
#         """
#         loss = self.pfy.apply(pred_cost, true_sol, self)
#         # reduction
#         if self.reduction == "mean":
#             loss = torch.mean(loss)
#         elif self.reduction == "sum":
#             loss = torch.sum(loss)
#         elif self.reduction == "none":
#             loss = loss
#         else:
#             raise ValueError("No reduction '{}'.".format(self.reduction))
#         return loss


# class perturbedFenchelYoungFunc(Function):
#     """
#     A autograd function for Fenchel-Young loss using perturbation techniques.
#     """

#     @staticmethod
#     def forward(ctx, pred_cost, true_sol, module):
#         """
#         Forward pass for perturbed Fenchel-Young loss

#         Args:
#             pred_cost (torch.tensor): a batch of predicted values of the cost
#             true_sol (torch.tensor): a batch of true optimal solutions
#             module (optModule): perturbedFenchelYoung module

#         Returns:
#             torch.tensor: solution expectations with perturbation
#         """
#         # get device
#         device = pred_cost.device
#         # convert tenstor
#         cp = pred_cost.detach().to("cpu").numpy()
#         w = true_sol.detach().to("cpu")
#         # sample perturbations
#         noises = module.rnd.normal(0, 1, size=(module.n_samples, *cp.shape))

#         ptb_c = cp + module.sigma * noises
#         ptb_c = ptb_c.reshape(-1, noises.shape[2])
#         # solve with perturbation
#         # ptb_sols, ptb_obj = _solve_or_cache(ptb_c, module)
#         module.optmodel.setObj(ptb_c)
#         ptb_sols, ptb_obj = module.optmodel.solve()

#         ptb_sols = ptb_sols.reshape(module.n_samples, -1, ptb_sols.shape[1])
#         # solution expectation
#         e_sol = ptb_sols.mean(axis=0)

#         # ptb_c = cp + module.sigma * noises
#         # solve with perturbation
#         # ptb_sols = _solve_or_cache(ptb_c, module)
#         # solution expectation
#         # e_sol = ptb_sols.mean(axis=1)
#         # difference
#         if module.optmodel.modelSense == EPO.MINIMIZE:
#             diff = w - e_sol
#         if module.optmodel.modelSense == EPO.MAXIMIZE:
#             diff = e_sol - w
#         # loss
#         loss = torch.sum(diff**2, axis=1)
#         # convert to tensor
#         diff = torch.FloatTensor(diff).to(device)
#         loss = torch.FloatTensor(loss).to(device)
#         # save solutions
#         ctx.save_for_backward(diff)
#         return loss

#     @staticmethod
#     def backward(ctx, grad_output):
#         """
#         Backward pass for perturbed Fenchel-Young loss
#         """
#         grad, = ctx.saved_tensors
#         grad_output = torch.unsqueeze(grad_output, dim=-1)
#         return grad * grad_output, None, None

# -------------------------------------------------------------------------
# Existing Loss Function Mapping
# -------------------------------------------------------------------------
# Registry mapping names to functions
LOSS_FUNCTIONS = {
    'SPO+': SPOPlus, # SPO Plus Loss
    'MSE': nn.MSELoss, # Mean Squared Error Loss
    'PG': PG_Loss, # PG loss
}

def get_loss_function(name: str) -> callable:
    """Utility function to get the loss function by name

    Args:
        name (str): name of the loss

    Returns:
        callable: loss function
    """
    if name not in LOSS_FUNCTIONS:
        raise ValueError(f"Loss function '{name}' not found. Available loss functions: {list(LOSS_FUNCTIONS.keys())}")
    return LOSS_FUNCTIONS[name]