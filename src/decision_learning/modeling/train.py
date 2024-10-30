import inspect

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from decision_learning.modeling.val_metrics import decision_regret

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


def init_loss_data_pretraining():
    pass


def train(
    pred_model: nn.Module,
    optmodel: nn.Module,
    loss_fn: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_metric: callable=decision_regret,
    device: str='cpu',
    optimizer: torch.optim.Optimizer=None,
    lr: float=1e-2,
    scheduler_params: dict={'step_size': 10, 'gamma': 0.1}):
    """The components needed to train in a decision-aware/focused manner:
    1. prediction model - for predicting the coefficients/parameters of the optimization model [done]
    2. optimization model/solver - for downstream decision-making task  
    3. loss function - to train pred model in a decision aware manner [done]
    4. training settings/parameters - optimizer to use, number of epochs, learning rate, lr schedule, gpu use, etc.
        - num_epochs
        - lr (learning rate)
        - optimizer
        - device
    5. dataloaders - to load data for training and validation. Assume that the original dataset object will use a 
    collate_fn function to create the batch as a dictionary, and 'X' will be the key for the features
        a. training dataloader - to load data for training the prediction model. Since each loss function will be highly 
            customized, the dataloader should be customized to the loss function. [done]
        b. val dataloader - data used to validate the model during training. Since each loss function will be highly 
            customized, the dataloader should be customized to the loss function. [done]
    7. val metric - metric to evaluate the model during training
    
    Args:
        pred_model (callable): _description_
    """
    
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
    # setup val metric - generally for decision-aware/focused, we care about the downstream optimization task, therefore
    # typically we'd want decision regret as val metric, but this requires an optimization model to be passed in so that
    # we can evaluate the decision solutions induced by the predicted cost. In this case we expect the val_metric to take
    # optmodel as an attribute.
    if 'optmodel' in inspect.signature(val_metric).parameters: # if function has optmodel as an argument
        val_metric = partial(val_metric, optmodel=optmodel) # set optmodel as an attribute of the function
    # log metrics
    metrics = []
    
    # training loop
    for epoch in range(num_epochs):
        
        # ------------------------- TRAINING -------------------------
        # THIS SECTION SHOULD NOT NEED TO BE MODIFIED - CUSTOM BEHAVIOR SHOULD BE SPECIFIED IN THE loss_fn FUNCTION
        epoch_losses = []
        pred_model.train() # beginning of epoch, set model to training mode
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            
            # move data to appropriate device. Assume that the collate_fn function in the dataset object will return a
            # dictionary with 'X' as the key for the features and remaining keys for other data required for specific loss fn
            for key in batch: 
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
        # TODO: finish validation regret metric
        pred_model.eval() # set model to evaluation mode
        val_losses = 0
        
        with torch.no_grad():            
            for batch_idx, batch in enumerate(tqdm(val_loader)):
                for key in batch:
                    batch[key] = batch[key].to(device)
                pred = pred_model(batch['X'])
                
                batch = filter_kwargs(val_metric, batch) # filter out only valid arguments for the val metric
                val_loss = val_metric(pred, **batch)
                val_losses += val_loss
        val_losses = val_losses / len(val_loader) # average val loss
        
        # ----------ADDITIONAL STEPS FOR OPTIMIZER/LOSS/MODEL ----------
        # MODIFY THIS SECTION IF YOU HAVE CUSTOM STEPS TO PERFORM EACH EPOCH LIKE SPECIAL PARAMETERS FOR THE LOSS FUNCTION
        # TODO: additional options - 1. early stopping, 2. model checkpointing
        # scheduler step
        if scheduler is not None:
            scheduler.step()
            
        # ------------ LOGGING AND REPORTING ---------------------------
        # MODIFY THIS SECTION IF YOU WANT TO LOG ADDITIONAL METRICS
        metrics.append({'epoch': epoch, 
                    'train_loss': np.mean(epoch_losses), 
                    'val_metric': val_losses})
        
        # TODO: add print statements to log the training progress
        
        
    metrics = pd.DataFrame(metrics)
    return metrics, pred_model