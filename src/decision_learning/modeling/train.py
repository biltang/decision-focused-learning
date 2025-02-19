import inspect
from functools import partial

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from decision_learning.modeling.val_metrics import decision_regret
from decision_learning.utils import filter_kwargs, log_runtime, handle_solver

# logging
import logging

logger = logging.getLogger(__name__)
logger.propagate = False
    
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Check if a stream handler already exists
if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
    # Create a stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    # Add the stream handler to the logger
    logger.addHandler(stream_handler)


class GenericDataset(Dataset):
    """Generic Dataset class for handling arbitrary input data. Useful for case where
    loss functions require highly customized data inputs, so user can specify arbitrary data in the form 
    of a named dictionary or as kwargs to GenericDataset class. Has following:
    - data: dictionary containing input data
    - length: number of samples in the dataset
    """
    
    def __init__(self, **kwargs):
        """Initializes the GenericDataset class with arbitrary input data.
        
        Args:
        - kwargs: dictionary containing input data
        """
        # Store all kwargs as attributes and convert to torch.tensor
        self.solver_kwargs = kwargs.pop('solver_kwargs', {})
        self.data = {key: torch.as_tensor(value, dtype=torch.float32) if not isinstance(value, torch.Tensor) else value
                     for key, value in kwargs.items()}
        
        # Determine the length of the dataset from one of the arguments
        self.length = len(next(iter(kwargs.values())))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Retrieve item for each key at the specified index
        item = {key: value[idx] for key, value in self.data.items()}
        item['solver_kwargs'] = {key: value[idx] for key, value in self.solver_kwargs.items()}
        return item
    
    
def init_loss_data_pretraining(data_dict: dict, 
                            dataloader_params: dict={'batch_size':32, 'shuffle':True}):
    """Wrapper function to convert user specified data_dict into a GenericDataset object and then into a DataLoader object.
    We require at a minimum the data_dict to contain the 'X' key for the features. The remaining keys can be arbitrary and should be
    specified based on the requirements of the loss function.

    Args:
        data_dict (dict): dictionary containing input data
        dataloader_params (dict, optional): data loader parameters. Defaults to {'batch_size':32, 'shuffle':True}.

    Returns:
        GenericDataset, DataLoader: GenericDataset object and DataLoader object
    """
    if 'X' not in data_dict:
        raise ValueError("data_dict must contain 'X' key")
    
    dataset = GenericDataset(**data_dict)
    dataloader = DataLoader(dataset, **dataloader_params)
    return dataset, dataloader


@log_runtime
def train(pred_model: nn.Module,
    optmodel: callable,
    loss_fn: nn.Module,
    train_data_dict: dict,
    val_data_dict: dict,
    test_data_dict: dict=None,
    dataloader_params: dict={'batch_size':32, 'shuffle':True},
    val_metric: callable=decision_regret,
    device: str='cpu',
    num_epochs: int=10,
    optimizer: torch.optim.Optimizer=None,
    lr: float=1e-2,
    scheduler_params: dict={'step_size': 10, 'gamma': 0.1},
    minimization: bool=True,
    verbose: bool=True,
    ):    
    """The components needed to train in a decision-aware/focused manner:
    1. prediction model - for predicting the coefficients/parameters of the optimization model [done]
    2. optimization model/solver - for downstream decision-making task  
    3. loss function - to train pred model in a decision aware manner [done]
    4. training settings/parameters - optimizer to use, number of epochs, learning rate, lr schedule, gpu use, etc.
        - num_epochs
        - lr (learning rate)
        - optimizer
        - device
    5. data - data used for training and validation. User needs to make sure the data is specified as a dictionary containing all the elements
        required for the loss function. [done]
        a. train_data_dict - data for training the prediction model. Since each loss function will be highly 
            customized, the dictionary should include data customized to the loss function. [done]
        b. val_data_dict - data used to validate the model during training. Since each loss function will be highly 
            customized, the dictionary should include data customized to the loss function. [done]
    7. val metric - metric to evaluate the model during training
    
    Args:
        pred_model (nn.Module): prediction model for predicting the coefficients/parameters of the optimization model
        optmodel (callable): optimization model/solver for downstream decision-making task
        loss_fn (nn.Module): loss function to train pred model in a decision aware manner
        train_data_dict (dict): data for training the prediction model
        val_data_dict (dict): data used to validate the model during training
        dataloader_params (dict, optional): parameters for the DataLoader. Defaults to {'batch_size':32, 'shuffle':True}.
        val_metric (callable, optional): metric to evaluate the model during training. Defaults to decision_regret.
        device (str, optional): device to use for training. Defaults to 'cpu'.
        num_epochs (int, optional): number of epochs to train the model. Defaults to 10.
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
        lr (float, optional): learning rate. Defaults to 1e-2.
        scheduler_params (dict, optional): parameters for the learning rate scheduler. Defaults to {'step_size': 10, 'gamma': 0.1}.
        minimization (bool, optional): whether the optimization task is a minimization task. Defaults to True.
        verbose (bool, optional): whether to print training progress. Defaults to True.                        
    """
    # ------------------------- SETUP -------------------------
    # training setup - setup things needed for training loop like optimizer and scheduler
    # optimizer
    if optimizer is None:
        optimizer = torch.optim.Adam(pred_model.parameters(), lr=lr)
        
    # scheduler
    scheduler = None
    if scheduler_params is not None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
        
    # set device related
    device = torch.device(device)
    pred_model.to(device)
    logger.info(f"Training on device: {device}")
    
    # setup val metric - generally for decision-aware/focused, we care about the downstream optimization task, therefore
    # we can evaluate the decision solutions induced by the predicted cost. In this case we expect the val_metric to take
    # typically we'd want decision regret as val metric, but this requires an optimization model to be passed in so that
    # optmodel as an attribute. (similar logic applies to other params we may want to pass in)
    preset_params = {'optmodel': optmodel, 'minimize': minimization}
    for param in preset_params.keys():
        if param not in inspect.signature(val_metric).parameters: 
            # if not in the function signature, drop it from the preset params
            preset_params.pop(param)
    val_metric = partial(val_metric, **preset_params) # set optmodel/minimize as an attribute of the function
            
    # log metrics
    metrics = []
    
    # DATA SETUP    
    train_dataset, train_loader = init_loss_data_pretraining(train_data_dict, dataloader_params)  # training data        
    val_dataset, val_loader = init_loss_data_pretraining(val_data_dict, dataloader_params)  # validation data
    
    
    # ------------------------- TRAINING LOOP -------------------------
    for epoch in range(num_epochs):
        
        # ------------------------- TRAINING -------------------------
        # THIS SECTION SHOULD NOT NEED TO BE MODIFIED - CUSTOM BEHAVIOR SHOULD BE SPECIFIED IN THE loss_fn FUNCTION
        epoch_losses = []
        pred_model.train() # beginning of epoch, set model to training mode
        for batch_idx, batch in enumerate(tqdm(train_loader, disable=not verbose, desc=f'Training Loader: Epoch {epoch+1}/{num_epochs}')):
            
            # move data to appropriate device. Assume that the collate_fn function in the dataset object will return a
            # dictionary with 'X' as the key for the features and remaining keys for other data required for specific loss fn
            for key in batch:
                if not isinstance(batch[key], dict): # don't move solver_kwargs to device since it is not important for pred_model and not a tensor
                    batch[key] = batch[key].to(device)
            
            # forward pass
            pred = pred_model(batch['X'])
            
            # -------- backwards pass --------
            # calculate loss. Since we assume loss_fn will be highly customized in terms of its inputs/function signature
            # we also assume batch contains all the necessary data required for the loss function, and we only need to supply the
            # current prediction. However, as a safety check, we will use helper function filter_kwargs to filter for only valid
            # arguments for the loss function to gracefully handle case where batch contains extra data not required by loss fn.
            batch = filter_kwargs(loss_fn.forward, batch) # filter out only valid arguments for the loss function                         
            loss = loss_fn(pred, **batch) # we assume the first argument of the loss function is the prediction, while the rest can be passed as kwargs
            
            # standard backwards pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())        
        
        # ------------------------- VALIDATION -------------------------
        # THIS SECTION SHOULD NOT NEED TO BE MODIFIED - CUSTOM BEHAVIOR SHOULD BE SPECIFIED IN THE val_metric FUNCTION
        pred_model.eval() # set model to evaluation mode
            
        # TODO: MODIFY VAL AND TEST REGRET CALC BEHAVIOR TO BE CONSISTENT
        
        # aggregate all predicted costs for entire validation set, then input into the val_metric function - assumption it all fits in memory
        val_regret = np.nan
        val_regret = calc_test_regret(pred_model=pred_model,
                                test_data_dict=val_data_dict,
                                optmodel=optmodel,
                                minimize=minimization                                
                            )                      
        
        # Test regret       
        test_regret = np.nan
        if test_data_dict is not None:
            test_regret = calc_test_regret(pred_model=pred_model,
                                test_data_dict=test_data_dict,
                                optmodel=optmodel,
                                minimize=minimization                                
                            )
        
        # ----------ADDITIONAL STEPS FOR OPTIMIZER/LOSS/MODEL ----------
        # MODIFY THIS SECTION IF YOU HAVE CUSTOM STEPS TO PERFORM EACH EPOCH LIKE SPECIAL PARAMETERS FOR THE LOSS FUNCTION
        # TODO: possible additional options - 1. early stopping, 2. model checkpointing
        # scheduler step
        if scheduler is not None:
            scheduler.step()
            
        # ------------ LOGGING AND REPORTING ---------------------------
        # MODIFY THIS SECTION IF YOU WANT TO LOG ADDITIONAL METRICS
        cur_metric = {'epoch': epoch, 
                    'train_loss': np.mean(epoch_losses), 
                    'val_metric': val_regret,
                    'test_regret': test_regret}
        metrics.append(cur_metric)
        
        if verbose:
            logger.info(f'epoch: {epoch+1}, train_loss: {np.mean(epoch_losses)}, val_metric: {val_regret}, test_regret: {test_regret}')

        
    metrics = pd.DataFrame(metrics)
    return metrics, pred_model


def calc_test_regret(pred_model: nn.Module, 
                    test_data_dict: dict, 
                    optmodel: callable, 
                    minimize: bool=True,                            
                    ):
    """Wrapper function to calculate regret of a given pred_model on a test set test_data_dict with optimization model optmodel.
    Decision (normalized) regret is calculated by calling function decision_regret from decision_learning.modeling.val_metrics 
    and is the difference between the cost of the predicted solution and the cost of the optimal solution.
    We need to supply:
        - pred: predicted costs of the test set based on features.
        - true_cost: the true cost of the test set
        - true_obj: the true objective value of the test set
    Therefore, test_data_dict must contain 'X', 'true_cost', and 'true_obj' keys with the corresponding values. 
    
    The function also serves as a wrapper to handle pytorch model prediction with torch.no_grad() context manager and assumes the test_data_dict
    and pred_model are small enough to fit in memory with only one batch.

    Args:
        pred_model (nn.Module): prediction model for predicting the coefficients/parameters of the optimization model
        test_data_dict (dict): dictionary containing input data for the test set. It must contain 'X', 'true_cost', and 'true_obj' keys.
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
        minimize (bool, optional): whether the optimization task is a minimization task. Defaults to True.

    Returns:
        float: decision regret of the pred_model on the test set
    """
    pred_model.eval()
    with torch.no_grad(): # context manager to disable gradient calculation
        X = torch.tensor(test_data_dict['X'], dtype=torch.float32)
        pred = pred_model(X)
        
        solver_kwargs = test_data_dict.get('solver_kwargs', {})      
        regret = decision_regret(pred,
                                true_cost=test_data_dict['true_cost'],
                                true_obj=test_data_dict['true_obj'],
                                optmodel=optmodel,
                                minimize=minimize,                                
                                solver_kwargs=solver_kwargs)
        
    return regret