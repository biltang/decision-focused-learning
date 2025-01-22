from torch import nn
import torch
from torch.autograd import Function
from torch.utils.data import Dataset
import numpy as np


# -------------------------------------------------------------------------
# SPO Plus (Smart Predict and Optimize Plus) Loss
# -------------------------------------------------------------------------

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
    
class perturbedFenchelYoung(nn.Module):
    """
    Wrapper function around custom perturbedFenchelYoungFunc with customized forwards, backwards pass. Extend
    from nn.Module to use nn.Module's functionalities.
    
    Autograd module for Fenchel-Young loss using perturbation techniques:
    Reference: <https://papers.nips.cc/paper_files/paper/2020/file/6bb56208f672af0dd65451f869fedfd9-Paper.pdf>    
    """
    def __init__(self, 
                optmodel: callable, 
                n_samples: int=10, 
                sigma: float=1.0, 
                seed: int=135,
                reduction: str="mean", 
                minimize: bool=True):
        """
        Args:
            optmodel (callable): a function/class that solves an optimization problem using pred_cost. For every batch of data, we use
                optmodel to solve the optimization problem using the predicted cost to get the optimal solution and objective value.
            n_samples (int): number of Monte-Carlo samples
            sigma (float): the amplitude of the perturbation
            seed (int): random state seed, since we are sampling for perturbation
            reduction (str): the reduction to apply to the output
            minimize (bool): whether the optimization problem is minimization or maximization            
        """
        super(perturbedFenchelYoung, self).__init__()
        
        self.pfy = perturbedFenchelYoungFunc()        
        self.n_samples = n_samples # number of samples        
        self.sigma = sigma # perturbation amplitude        
        self.rnd = np.random.RandomState(seed) # random state        
        self.reduction = reduction
        self.minimize = minimize
        self.optmodel = optmodel        
        

    def forward(self, pred_cost: torch.tensor, true_sol: torch.tensor):
        """
        Forward pass
        
        Args:
            pred_cost (torch.tensor): a batch of predicted values of the cost
            true_sol (torch.tensor): a batch of true optimal solutions
        """
        loss = self.pfy.apply(pred_cost, 
                            true_sol,
                            self.n_samples,
                            self.rnd,
                            self.optmodel,
                            self.sigma,
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


class perturbedFenchelYoungFunc(Function):
    """
    A autograd function for Fenchel-Young loss using perturbation techniques.
    """

    @staticmethod
    def forward(ctx, 
            pred_cost: torch.tensor, 
            true_sol: torch.tensor,             
            n_samples: int,
            sampler: np.random.RandomState,
            optmodel: callable,
            sigma: float=1.0,                         
            minimize: bool=True):        
        """
        Forward pass for perturbed Fenchel-Young loss

        Args:
            pred_cost (torch.tensor): a batch of predicted values of the cost
            true_sol (torch.tensor): a batch of true optimal solutions
            n_samples (int): number of Monte-Carlo samples
            sigma (float): the amplitude of the perturbation
            sampler (np.random.RandomState): random state for sampling perturbations
            optmodel (callable): a function/class that solves an optimization problem using pred_cost. For every batch of data, we use
                optmodel to solve the optimization problem using the predicted cost to get the optimal solution and objective value.
            minimize (bool): whether the optimization problem is minimization or maximization

        Returns:
            torch.tensor: solution expectations with perturbation
        """
        
        # TODO: add support for detach and convert to numpy since optmodel may not be written for torch tensors
        # detach (stops gradient tracking since we will compute custom gradient) and move to cpu. Do this since
        # generally the optmodel is probably cpu based
        
        # rename variable names for convenience
        # cp for predicted cost, w for true solution (based on true costs)
        cp = pred_cost.detach().to("cpu")
        w = true_sol.detach().to("cpu")
        
        # for FYL loss, we use mont-carlo sampling to create perturbations around the predicted cost. The perturbations
        # are sampled from a normal distribution with mean 0 and standard deviation sigma. We then solve the optimization using perturbed costs,
        # which induces perturbed solutions that are not locally constant and differentiatiable.
        noises = sampler.normal(0, 1, size=(n_samples, *cp.shape)) 
        # noise is shape (n_sample, batch_size (cp.shape[0]), cost vector dimension size (cp.shape[1])), so that 
        # noise[i, :] is the i-th perturbation sample for the entire batch.
        
        # use broadcasting to add scaled noises (n_samples, batch_size, cost_vector_dim) to the predicted cost (batch_size, cost_vector_dim),
        # so that for each perturbed sample noise[i, :] (batch_size, cost_vector_dim), the predicted cost is added to it since they are same shape.
        # end result is ptb_c is shape (n_samples, batch_size, cost_vector_dim) where ptb_c[i, :] is the i-th perturbed cost sample for the entire batch.        
        ptb_c = cp + sigma * noises # scale noises matrix with sigma
        
        # reshape from (n_samples, batch_size, cost_vector_dim) to (n_samples * batch_size, cost_vector_dim). This is because we need to plug each
        # ptb_c[i, j, :] (ith perturbation, jth batch sample) into the optimization problem to get the optimal solution and objective value, and the optimization problem
        # is solved for a given cost vector sample.
        ptb_c = ptb_c.reshape(-1, noises.shape[2]) 
        
        # solve optimization problem to obtain optimal sol/obj val from perturbed costs (based on predicted costs), 
        # where now ptb_c[k, :] is k = i*j example that is the ith perturbed cost sample for the jth batch sample
        ptb_sols, ptb_obj = optmodel(ptb_c)
        
        # reshape back to (n_samples, batch_size, sol_vector_dim) where ptb_sols[i, j, :] is the ith perturbed solution sample for the jth batch sample to get back to original data shape
        ptb_sols = ptb_sols.reshape(n_samples, -1, ptb_sols.shape[1])
        
        # expected/average solution - expectation over perturbed samples
        exp_sol = ptb_sols.mean(axis=0) # axis=0 is the n_samples axis, so we get the expectation over n_sample perturbed samples for each original batch sample
  
        # save solutions for backwards pass
        ctx.save_for_backward(exp_sol, w)
        ctx.minimize = minimize
        
        loss = torch.sum((w - exp_sol)**2, axis=1)
        if not minimize:
            loss = -loss
  
        loss = torch.FloatTensor(loss)
        return loss
        
        
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for perturbed Fenchel-Young loss
        """
        exp_sol, w = ctx.saved_tensors
        
        grad = w - exp_sol
        if not ctx.minimize:
            grad = - grad
                
        grad_output = torch.unsqueeze(grad_output, dim=-1)
        return grad * grad_output, None, None, None, None, None, None, None


# -------------------------------------------------------------------------
# Cosine Surrogates
# -------------------------------------------------------------------------
class CosineSurrogateDotProdMSE(nn.Module):
    """Implements a convexified surrogate loss function for cosine similarity loss by taking
    a linear combination of the mean squared error (MSE) and the dot product of the predicted and true costs since
    - MSE captures magnitude of the difference between predicted and true costs
    - Dot product captures the direction/angle of the difference between predicted and true costs
    """
    
    def __init__(self, alpha: float=1, reduction: str='mean', minimize: bool=True):
        """
        Args:
            alpha (float, optional): Weighting parameter for how heavily to weigh MSE component of loss vs dot product. Defaults to 1.
            reduction (str): the reduction to apply to the output. Defaults to 'mean'.
            minimize (bool): whether the optimization problem is minimization or maximization. Defaults to True.
        """
        super(CosineSurrogateDotProdMSE, self).__init__()
        self.alpha = alpha
        self.reduction = reduction
        self.minimize = minimize
        self.mse_loss = nn.MSELoss(reduction=reduction) # use off-the-shelf MSE loss
        

    def forward(self, pred_cost: torch.tensor, true_cost: torch.tensor):
        """Takes the predicted and true costs and computes the loss using the convexified cosine surrogate loss function
        that is linear combination of MSE and dot product of predicted and true costs.
        
        Args:
            pred_cost (torch.tensor): a batch of predicted values of the cost
            true_cost (torch.tensor): a batch of true values of the cost        
        """        
        mse = self.mse_loss(pred_cost, true_cost)
        
        # ----- Compute dot product -----
        dot_product = torch.sum(pred_cost * true_cost, dim=1)
        if self.minimize:
            dot_product = -dot_product # negate dot product for minimization
            
        # reduction
        if self.reduction == "mean":
            dot_product = torch.mean(dot_product)
        elif self.reduction == "sum":
            dot_product = torch.sum(dot_product)
        elif self.reduction == "none":
            dot_product = dot_product
        else:
            raise ValueError("No reduction '{}'.".format(self.reduction))
        
        loss = self.alpha * mse + dot_product  # compute final loss as linear combination of MSE and dot product        
       
        return loss
    
    
class CosineSurrogateDotProdVecMag(nn.Module):
    """Implements a convexified surrogate loss function for cosine similarity loss by taking
    trying to maximize the dot product of the predicted and true costs while simultaneously minimizing the magnitude of the predicted cost
    since this would incentivize the predicted cost to be in the same direction as the true cost without the predictions artificially
    making the dot product higher by increasing the magnitude of the predicted cost.    
    """
    def __init__(self, alpha: float=1, reduction: str='mean', minimize: bool=True):
        """
        Args:
            alpha (float, optional): Weight emphasis on minimizing magnitude of predicted vector (measured through self dot product). Defaults to 1.
            reduction (str): the reduction to apply to the output. Defaults to 'mean'.
            minimize (bool): whether the optimization problem is minimization or maximization. Defaults to True.
        """
        super(CosineSurrogateDotProdVecMag, self).__init__()
        self.alpha = alpha
        self.reduction = reduction
        self.minimize = minimize
        

    def forward(self, pred_cost: torch.tensor, true_cost: torch.tensor):
        """Computes the loss using a linear combination of two components:
        1) self dot product - measures the magnitude of the predicted cost vector, trying to minimize it
        2) dot product of predicted and true costs - measures the direction of the predicted cost vector, trying to maximize it

        Args:
            pred_cost (torch.tensor): a batch of predicted values of the cost
            true_cost (torch.tensor): a batch of true values of the cost          
        """    
        dot_product_self = torch.sum(pred_cost * pred_cost, dim=1)
        
        # dot product of predicted and true costs - measure angle between predicted and true costs
        dot_product_ang = torch.sum(pred_cost * true_cost, dim=1)
        if self.minimize:
            dot_product_ang = -dot_product_ang # negate dot product for minimization
        
        loss = self.alpha * dot_product_self + dot_product_ang  # compute final loss as linear combination of self dot product and dot product        
        
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
    
        
# -------------------------------------------------------------------------
# Existing Loss Function Mapping
# -------------------------------------------------------------------------
# Registry mapping names to functions
LOSS_FUNCTIONS = {
    'SPO+': SPOPlus, # SPO Plus Loss
    'MSE': nn.MSELoss, # Mean Squared Error Loss
    'Cosine': nn.CosineEmbeddingLoss, # Cosine Embedding Loss
    'PG': PG_Loss, # PG loss
    'FYL': perturbedFenchelYoung, # perturbed Fenchel-Young loss
    'CosineSurrogateDotProdMSE': CosineSurrogateDotProdMSE, # Cosine Surrogate Dot Product MSE Loss
    'CosineSurrogateDotProdVecMag': CosineSurrogateDotProdVecMag # Cosine Surrogate Dot Product Vector Magnitude Loss
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