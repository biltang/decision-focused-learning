import importlib
from functools import partial
import time
import os

from torch import nn
import torch
import pandas as pd

import decision_learning.modeling.pipeline
import decision_learning.data.shortest_path_grid
from decision_learning.utils import handle_solver
from decision_learning.modeling.models import LinearRegression
from decision_learning.modeling.pipeline import lossfn_experiment_pipeline, lossfn_hyperparam_grid
from decision_learning.data.shortest_path_grid import genData

# logging
import logging
logging.basicConfig(level=logging.INFO)

def shortest_path_solver(costs, size, sens = 1e-4):
    if isinstance(size, np.ndarray):
        size = int(size[0])   
    elif isinstance(size, torch.Tensor):     
        size = int(size[0].item())
    
    if type(size) != int:
        size = int(size)
        
    # Forward Pass
    starting_ind = 0
    starting_ind_c = 0
    samples = costs.shape[0]
    V_arr = torch.zeros(samples, size ** 2)
    for i in range(0, 2 * (size - 1)):
        num_nodes = min(i + 1, 9 - i)
        num_nodes_next = min(i + 2, 9 - i - 1)
        num_arcs = 2 * (max(num_nodes, num_nodes_next) - 1)
        V_1 = V_arr[:, starting_ind:starting_ind + num_nodes]
        layer_costs = costs[:, starting_ind_c:starting_ind_c + num_arcs]
        l_costs = layer_costs[:, 0::2]
        r_costs = layer_costs[:, 1::2]
        next_V_val_l = torch.ones(samples, num_nodes_next) * float('inf')
        next_V_val_r = torch.ones(samples, num_nodes_next) * float('inf')
        if num_nodes_next > num_nodes:
            next_V_val_l[:, :num_nodes_next - 1] = V_1 + l_costs
            next_V_val_r[:, 1:num_nodes_next] = V_1 + r_costs
        else:
            next_V_val_l = V_1[:, :num_nodes_next] + l_costs
            next_V_val_r = V_1[:, 1:num_nodes_next + 1] + r_costs
        next_V_val = torch.minimum(next_V_val_l, next_V_val_r)
        V_arr[:, starting_ind + num_nodes:starting_ind + num_nodes + num_nodes_next] = next_V_val

        starting_ind += num_nodes
        starting_ind_c += num_arcs

    # Backward Pass
    starting_ind = size ** 2
    starting_ind_c = costs.shape[1]
    prev_act = torch.ones(samples, 1)
    sol = torch.zeros(costs.shape)
    for i in range(2 * (size - 1), 0, -1):
        num_nodes = min(i + 1, 9 - i)
        num_nodes_next = min(i, 9 - i + 1)
        V_1 = V_arr[:, starting_ind - num_nodes:starting_ind]
        V_2 = V_arr[:, starting_ind - num_nodes - num_nodes_next:starting_ind - num_nodes]

        num_arcs = 2 * (max(num_nodes, num_nodes_next) - 1)
        layer_costs = costs[:, starting_ind_c - num_arcs: starting_ind_c]

        if num_nodes < num_nodes_next:
            l_cs_res = ((V_2[:, :num_nodes_next - 1] - V_1 + layer_costs[:, ::2]) < sens) * prev_act
            r_cs_res = ((V_2[:, 1:num_nodes_next] - V_1 + layer_costs[:, 1::2]) < sens) * prev_act
            prev_act = torch.zeros(V_2.shape)
            prev_act[:, :num_nodes_next - 1] += l_cs_res
            prev_act[:, 1:num_nodes_next] += r_cs_res
        else:
            l_cs_res = ((V_2 - V_1[:, :num_nodes - 1] + layer_costs[:, ::2]) < sens) * prev_act[:, :num_nodes - 1]
            r_cs_res = ((V_2 - V_1[:, 1:num_nodes] + layer_costs[:, 1::2]) < sens) * prev_act[:, 1:num_nodes]
            prev_act = torch.zeros(V_2.shape)
            prev_act += l_cs_res
            prev_act += r_cs_res
        cs = torch.zeros(layer_costs.shape)
        cs[:, ::2] = l_cs_res
        cs[:, 1::2] = r_cs_res
        sol[:, starting_ind_c - num_arcs: starting_ind_c] = cs

        starting_ind = starting_ind - num_nodes
        starting_ind_c = starting_ind_c - num_arcs
    # Dimension (samples, num edges)
    obj = torch.sum(sol * costs, axis=1)
    # Dimension (samples, 1)
    sol = sol.to(torch.float32)
    obj = obj.reshape(-1,1).to(torch.float32)
    return sol, obj


def main():
    
    # ----------------------- SETUP -----------------------
    
    # ----------- Data Setup -----------
    # set random seeds
    torch.manual_seed(105)
    indices_arr = torch.randperm(100000)
    indices_arr_test = torch.randperm(100000)

    # construct experiment data configuration array
    # we'll run 100 trials for each configuration of n and epsilon type
    n_arr = [200, 400, 800, 1600] # number of training data
    ep_arr = ['unif', 'normal'] # epsilon type
    trials = 100

    # generate experiment array
    exp_arr = []
    for n in n_arr:
        for ep in ep_arr:
            for t in range(trials):
                exp_arr.append([n, ep, t])

    # shortest path example data generation configurations
    grid = (5, 5)  # grid size    
    num_feat = 5  # size of feature
    deg = 6  # polynomial degree
    e = .3  # noise width
    
    # path planting for shortest path example - see page 9, subsection "Harder Example with Planted Arcs" in section 4.2 of paper https://arxiv.org/pdf/2402.03256
    planted_good_pwl_params = {'slope0':0, # slope of first segment of piecewise linear cost function for "good" edge cost planted
                        'int0':2, # intercept of first segment of piecewise linear cost function for "good" edge cost planted
                        'slope1':0, # slope of second segment of piecewise linear cost function for "good" edge cost planted
                        'int1':2} # intercept of second segment of piecewise linear cost function for "good" edge cost planted
    planted_bad_pwl_params = {'slope0':4, # slope of first segment of piecewise linear cost function for "bad" edge cost planted
                        'int0':0, # intercept of first segment of piecewise linear cost function for "bad" edge cost planted
                        'slope1':0, # slope of second segment of piecewise linear cost function for "bad" edge cost planted
                        'int1':2.2} # intercept of second segment of piecewise linear cost function for "bad" edge cost planted
    plant_edge = True # to plant edges in shortest path experiment or not
    
    # ----------- optimization model -----------
    optmodel = partial(handle_solver, optmodel=shortest_path_solver, detach_tensor=False, solver_batch_solve=True)

    # ----------- Results Setup -----------
    save_file = 'shortest_path_neurips_exp_opthandler_check.csv'
    
    # ----------------------- Sequentially loop through experiment trials -----------------------
    # experiment does not take that long (< 10 hours), so we can run it sequentially on HPC instead of parallel processing across multiple compute nodes
         
    for sim in range(len(exp_arr)): 
        start_time = time.time()
        # setup
        exp = exp_arr[sim] # current experiment from experiment array
        num_data = exp[0]  # number of training data
        ep_type = exp[1] # noise type of current experiment
        trial = exp[2] # trial number of current experiment
        logging.info(f"Running experiment: {sim+1}/{len(exp_arr)} with n={num_data}, epsilon type={ep_type}, trial={trial}")
        
        # ------------ Instance specific data ------------
        # training data
        generated_data = genData(num_data=num_data+200, # number of data points to generate for training set
                num_features=num_feat, # number of features 
                grid=grid, # grid shape
                deg=deg, # polynomial degree
                noise_type=ep_type, # epsilon noise type
                noise_width=e, # amount of noise
                seed=indices_arr[trial], # seed the randomness
                plant_edges=plant_edge, # to plant edges or not
                planted_good_pwl_params=planted_good_pwl_params, # cost function for good edges
                planted_bad_pwl_params=planted_bad_pwl_params) # cost function for bad edges

        # testing data
        generated_data_test = genData(num_data=10000, # number of data points to generate for test set
                num_features=num_feat, # number of features 
                grid=grid,  # grid shape
                deg=deg,  # polynomial degree
                noise_type=ep_type,  # epsilon noise type
                noise_width=e, # amount of noise
                seed=indices_arr_test[trial],      # seed the randomness
                plant_edges=plant_edge, # to plant edges or not
                planted_good_pwl_params=planted_good_pwl_params, # cost function for good edges
                planted_bad_pwl_params=planted_bad_pwl_params) # cost function for bad edges
    
        # ------------prediction model------------
        pred_model = LinearRegression(input_dim=generated_data['feat'].shape[1],
                        output_dim=generated_data['cost'].shape[1])
    
        # ------------loss function experiment pipeline------------
        
        non-PG losses
        preimplement_loss_results, preimplement_loss_models = lossfn_experiment_pipeline(X_train=generated_data['feat'],
                true_cost_train=generated_data['cost'],
                X_test=generated_data_test['feat'],
                true_cost_test=generated_data_test['cost_true'], 
                predmodel=pred_model,
                optmodel=optmodel,
                val_split_params={'test_size':200, 'random_state':42},
                loss_names=['SPO+', 'MSE', 'FYL', 'Cosine'],                            
                training_configs={'num_epochs':100,
                                 'dataloader_params': {'batch_size':32, 'shuffle':True}},
                save_models=True                                                                                 
                )
        
        # PG loss
        PG_init_results, PG_init_models = lossfn_experiment_pipeline(X_train=generated_data['feat'],
                true_cost_train=generated_data['cost'],
                X_test=generated_data_test['feat'],
                true_cost_test=generated_data_test['cost_true'], 
                predmodel=preimplement_loss_models['SPO+_{}'],
                optmodel=optmodel,
                val_split_params={'test_size':200, 'random_state':42},
                loss_names=['PG'],
                loss_configs={'PG': {'h':[num_data**-.125, num_data**-.25, num_data**-.5, num_data**-1], 'finite_diff_type': ['B', 'C', 'F']}},
                training_configs={'num_epochs':100,
                                 'dataloader_params': {'batch_size':32, 'shuffle':True}},
                save_models=False
                )

        # Cosine Surrogate Losses        
        cos_surr_results, cos_surr_models = lossfn_experiment_pipeline(X_train=generated_data['feat'],
                        true_cost_train=generated_data['cost'],
                        X_test=generated_data_test['feat'],
                        true_cost_test=generated_data_test['cost_true'], 
                        predmodel=pred_model,
                        optmodel=optmodel,
                        val_split_params={'test_size':200, 'random_state':42},
                        loss_names=['CosineSurrogateDotProdVecMag','CosineSurrogateDotProdMSE'],
                        loss_configs={'CosineSurrogateDotProdVecMag': {'alpha':[0.01, 0.1, 1, 2.5, 5, 7.5, 10]},
                                      'CosineSurrogateDotProdMSE': {'alpha':[0.01, 0.1, 1, 2.5, 5, 7.5, 10]}},
                        training_configs={'num_epochs':100,
                                         'dataloader_params': {'batch_size':32, 'shuffle':True}},
                        save_models=False
                        )


        # store current results
        combined_results = cos_surr_results #pd.concat([preimplement_loss_results, PG_init_results], ignore_index=True)
        combined_results['n'] = num_data
        combined_results['eps_type'] = ep_type
        combined_results['trial'] = trial
    
        # write results to file
        if os.path.exists(save_file):
            combined_results.to_csv(save_file, mode='a', header=False, index=False)
            logging.info(f"Appended to {save_file}")
        else:
            combined_results.to_csv(save_file, mode='w', header=True, index=False)
            logging.info(f"Created {save_file}")
            
        # display time taken per trial
        end_time = time.time()
        logging.info(f"Time taken for experiment: {end_time - start_time} seconds")
            

if __name__ == "__main__":
    main()