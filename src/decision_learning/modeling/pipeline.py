from typing import List, Union
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
    
    
def lossfn_experiment_data_pipeline(X: Union[np.ndarray, torch.tensor], 
                                true_cost: Union[np.ndarray, torch.tensor], 
                                optmodel: callable):
    """Wrapper function to preprocess data for experiments on loss functions implemented within the code base in decision_learning.modeling.loss
    Since decision-aware/focused problems generally compare the optimal solution/obj under the true_cost vs the solution/obj under the predicted cost,
    we precompute the optimal solution and objective under the true cost as "true_sol" and "true_obj". For flexibility reasons, the code base/train/pipeline/loss functions
    expect data to be passed as dictionaries with key, value pairs where the key names match up to the input arguments into loss functions. 
    
    This function also wraps the data into a dictionary with the keys "X", "true_cost", "true_sol", "true_obj" for consistency across loss functions.
    
    Args:
        X (Union[np.ndarray, torch.tensor]): features
        true_cost (Union[np.ndarray, torch.tensor]): true cost
        optmodel (callable): optimization model that takes in true_cost and returns optimal solution and objective

    Returns:
        dict: dictionary with keys "X", "true_cost", "true_sol", "true_obj" for consistency across loss functions
    """
    
    sol, obj = optmodel(true_cost) # get optimal solution and objective under true cost
    final_data = {"X": X, "true_cost": true_cost, "true_sol": sol, "true_obj": obj} # wrap data into dictionary
    return final_data


def existing_lossfn_data_preprocess(loss_name: str, data_dict: dict):
    """Each loss function implemented in decision_learning.modeling.loss may have specific inputs it expects beyond the generic "X", "true_cost", "true_sol", "true_obj" data.
    This function is essentially a switch/lookup case to modify the data_dict to match the expected input arguments of the loss function. This is expected to be called before training
    and only works for existing loss functions implemented in the code base.

    Args:
        loss_name (str): name of the loss function
        data_dict (dict): dictionary with keys "X", "true_cost", "true_sol", "true_obj" for consistency across loss functions

    Returns:
        dict: modified data_dict with keys matching the expected input arguments of the loss function
    """
    # simple check loss_name and modify data_dict accordingly to make sure inputs will be the argument names expected
    if loss_name == "MSE":
        data_dict['target'] = data_dict['true_cost'] # nn.MSE takes target argument as the true label to be predicted, which is the true cost 
    
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


def lossfn_experiment_pipeline(X_train: Union[np.ndarray, torch.tensor], 
            true_cost_train: Union[np.ndarray, torch.tensor], 
            X_test: Union[np.ndarray, torch.tensor],
            true_cost_test: Union[np.ndarray, torch.tensor],
            predmodel: callable,
            optmodel: callable,
            val_split_params: dict={'test_size':0.2, 'random_state':42},
            loss_names: List[str]=[], 
            loss_configs: dict={}, 
            custom_loss_inputs: List[dict]=[],
            minimize: bool=True,
            training_configs: dict=None,
            save_models: bool=False):
    """High level function to run an experiment pipeline for decision-aware/focused learning.

    Args:
        X_train (Union[np.ndarray, torch.tensor]): training features
        true_cost_train (Union[np.ndarray, torch.tensor]): training true cost
        X_test (Union[np.ndarray, torch.tensor]): test features
        true_cost_test (Union[np.ndarray, torch.tensor]): test true cost
        predmodel (callable): pytorch prediction model
        optmodel (callable): optimization model that takes in true_cost and returns optimal solution and objective
        val_split_params (dict, optional): how to split training data into train/val splits. Defaults to {'test_size':0.2, 'random_state':42}.
        loss_names (List[str], optional): list of loss functions to run experiment pipeline on that are implemented already in the codebase in decision_learning.modeling.loss. Defaults to [].
        
        loss_configs (dict, optional): dictionary mapping from loss_name (key) to a dictionary of different hyperparameters that are then grid searched over. 
            Ex: {'PG': {'h':[num_data**-.125, num_data**-.25, num_data**-.5, num_data**-1], 'finite_diff_type': ['B', 'C', 'F']}}
            The assumption is the hyperparameter grid generated per loss function would not be too big. Defaults to {}.
        custom_loss_inputs (List[dict], optional): list of custom loss function configurations to run through the train function as part of experient pipeline. Because it is user custom,
            user is expected to provide inputs in the form:
            {'loss_name': name of the loss function, 
            'loss': a callable loss function,
            'data': dictionary with the proper argument names expected by the loss function}. 
            Example:
             [{'loss_name':'cosine',
               'loss':nn.CosineEmbeddingLoss,
               'data': {'X': generated_data['feat'],
                        'input2':generated_data['cost'], 
                        'target':torch.ones(generated_data['cost'].shape[0])}
                       }
            ].
            Defaults to [].
        minimize (bool, optional): minimization problem?. Defaults to True.
        training_configs (dict, optional): parameters to be passed into train function for pytorch training loop. 
            Example:
            {
                'dataloader_params': {'batch_size':32, 'shuffle':True},
                'num_epochs': 10,
                'lr': 0.01,
                'scheduler_params': None
            }. 
            Defaults to None.        
        save_models (bool, optional): flag to save models or not. If we are searching over many hyperparameters/loss function/models, 
                                    may be impractical to store all of them since we may not be able to fit it in all in memory. 
                                    However, this may be useful for storing select models in memory that can then be used as initialization points for other
                                    loss function/model experiments. Defaults to False.

    Raises:
        ValueError: if no loss function is provided

    Returns:
        pd.DataFrame, dict: overall_metrics, trained_models
    """
    
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
            # TODO: fix logging to find right level of detail to output and way for user to control the logging level
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
            
            metrics, trained_model = train(pred_model=pred_model, # call train function
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
            
            if save_models: # if save_models, store trained_model under loss_name and hyperparameters                
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
        
        if save_models: # if save_models, store trained_model under loss_name and hyperparameters
            trained_models[custom_loss_input['loss_name']] = trained_model
        
    overall_metrics = pd.concat(overall_metrics, ignore_index=True)
    return overall_metrics, trained_models