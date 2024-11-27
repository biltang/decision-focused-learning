from typing import List
from itertools import product
import copy

from sklearn.model_selection import train_test_split
import pandas as pd

from decision_learning.modeling.loss import get_loss_function
from decision_learning.utils import filter_kwargs
from decision_learning.modeling.train import train

# logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Check if a stream handler already exists
if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
    # Create a stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    # Add the stream handler to the logger
    logger.addHandler(stream_handler)
    
    
def lossfn_experiment_data_pipeline(X, true_cost, optmodel: callable):
    
    sol, obj = optmodel(true_cost)
    final_data = {"X": X, "true_cost": true_cost, "true_sol": sol, "true_obj": obj}
    return final_data


def existing_lossfn_data_preprocess(loss_name: str, data_dict: dict):
    
    # simple check loss_name and modify data_dict accordingly to make sure inputs will be the argument names expected
    if loss_name == "MSE":
        data_dict['target'] = data_dict['true_cost']
    
    return data_dict 


def lossfn_hyperparam_grid(hyperparams: dict[str, list]) -> list[dict]:
    """Create all possible combinations of hyperparameters from a dictionary of lists.

    Args:
        hyperparams (dict): dictionary of hyperparameters, where key is the hyperparameter name and value is a list of values to try

    Returns:
        list[dict]: list of hyperaparameter combinations, where each combination is a dictionary of parameter name and its value to try
    """
    # Extract parameter names and corresponding value lists
    param_names = hyperparams.keys()
    param_values = hyperparams.values()
    
    # Use itertools.product to get all combinations
    combinations = list(product(*param_values))
    
    # Convert each combination (tuple) into a dictionary
    return [dict(zip(param_names, combination)) for combination in combinations] 


def lossfn_experiment_pipeline(X_train,
            true_cost_train,
            X_test,
            true_cost_test, 
            predmodel: callable,
            optmodel: callable,
            val_split_params: dict={'test_size':0.2, 'random_state':42},
            loss_names: List[str]=[], 
            loss_configs: dict={}, 
            custom_loss_inputs: List[dict]=[],
            minimize: bool=True,
            training_configs: dict=None,
            save_models: bool=False):
        
    # default training configs
    # set up this way so that we have a default configuration for training, but can override it with user provided configs
    # dynamically thru dictionary update function, where now only user provided keys will be updated
    tr_config = {
        'dataloader_params': {'batch_size':32, 'shuffle':True},
        'num_epochs': 10,
        'lr': 0.01,
        'scheduler_params': None
        }
    if training_configs is not None:
        tr_config.update(training_configs)
        
    # check if a loss function is provided (either off the shelf or custom)
    if not loss_names and custom_loss_inputs is None:
        raise ValueError("Please provide a loss function")
    
    # -----------------EXISTING LOSS FUNCTIONS: GET BY NAME PROVIDED IN loss_names----------------- 
    
    # -----------------Initial data preprocessing for existing loss functions-----------------
    # This is done to ensure that the data is in the correct format for the loss functions
    # training data
    train_d = lossfn_experiment_data_pipeline(X_train, true_cost_train, optmodel)
    
    # split train/val data
    # Splitting each input in the same way
    train_dict = {}
    val_dict = {}

    for key, value in train_d.items():
        train_data, val_data = train_test_split(value, **val_split_params)
        train_dict[key] = train_data
        val_dict[key] = val_data
        
    # testing data
    test_data = lossfn_experiment_data_pipeline(X_test, true_cost_test, optmodel)
    
    # -----------------EXPERIMENT LOGGING SETUP----------------- 
    overall_metrics = []
    trained_models = {}
    # loop through list of existing loss functions
    for loss_n in loss_names: 
        cur_loss_fn = get_loss_function(loss_n)
        
        # loss function parameters
        cur_loss_fn_hyperparam_grid = [{}] # by default, no hyperparameters, in which case, **{} is equivalent to using default values
        if loss_n in loss_configs:
            cur_loss_fn_hyperparam_grid = lossfn_hyperparam_grid(loss_configs[loss_n])
        #logger.debug(f"Loss name {loss_n}, function {cur_loss_fn}, and Loss function hyperparameters grid: {cur_loss_fn_hyperparam_grid}")
        
        # loop over possible cur_loss_fn_hyperparam_grid, if none provided within loss_configs, 
        # then only one iteration, where we pass {} to the loss function, which results in default values
        for param_set in cur_loss_fn_hyperparam_grid:
            
            # copy the param_set to avoid modifying the original dictionary
            orig_param_set = copy.deepcopy(param_set)
            
            # additional params to add to param_set - optmodel, minimization, etc.
            additional_params = {"optmodel": optmodel, "minimize": minimize}
            param_set.update(additional_params)
            
            # filter out additional params that are not needed by the loss function
            param_set = filter_kwargs(func=cur_loss_fn.__init__, kwargs=param_set)
            #logger.debug(f"Filtered param set: {param_set} input into loss function {cur_loss_fn}")
            
            # instantiate the loss function
            cur_loss = cur_loss_fn(**param_set) # instantiate the loss function - optionally with configs if provided            
            
            # ADDITIONAL: create correct data input for off-the-shelf loss/preexisting loss function
            train_dict = existing_lossfn_data_preprocess(loss_name=loss_n, data_dict=train_dict)
            
            # -----------------TRAINING LOOP-----------------
            # TODO: decide if all loss functions should start from the same model 
            # (in which case copy the model to avoid modifying the original model),
            # or if just random initialization each time. Currently just deep copy the model
            # TODO: allow for user initialization of prediciton model
            pred_model = copy.deepcopy(predmodel)
            
            metrics, trained_model = train(pred_model=pred_model,
                optmodel=optmodel,
                loss_fn=cur_loss,
                train_data_dict=train_dict,
                val_data_dict=val_dict,
                test_data_dict=test_data,
                minimization=minimize,
                **tr_config)
            metrics['loss_name'] = loss_n            
            metrics['hyperparameters'] = str(orig_param_set)
            
            overall_metrics.append(metrics)
            
            if save_models:                
                trained_models[loss_n + "_" + str(orig_param_set)] = trained_model
            
    # -----------------TODO: CUSTOM LOSS FUNCTION: GET BY NAME PROVIDED IN custom_loss_inputs-----------------
    # TODO: check if names match up with custom loss functions - raise Error otherwise
    for custom_loss_input in custom_loss_inputs:
        
        cur_loss = custom_loss_input['loss']()
        
        # TODO: add functionality to also search over a a grid of hyperparameters for custom loss functions
        pred_model = copy.deepcopy(predmodel)
            
        metrics, trained_model = train(pred_model=pred_model,
            optmodel=optmodel,
            loss_fn=cur_loss,
            train_data_dict=custom_loss_input['data'],
            val_data_dict=val_dict,
            test_data_dict=test_data,
            minimization=minimize,
            **tr_config)
        metrics['loss_name'] = custom_loss_input['loss_name']
        metrics['hyperparameters'] = None            
        overall_metrics.append(metrics)
        
        if save_models:
            trained_models[custom_loss_input['loss_name']] = trained_model
        
    overall_metrics = pd.concat(overall_metrics, ignore_index=True)
    return overall_metrics, trained_models