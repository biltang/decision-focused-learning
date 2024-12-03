from typing import Tuple

import numpy as np
import torch 

# Create a logger
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Check if any handlers exist
if not logger.handlers:
    # Create a stream handler
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler) # Add the stream handler to the logger


def piecewise_linear(x: np.ndarray, 
                    slope0: float, 
                    int0: float,
                    slope1: float, 
                    int1: float, 
                    scaling: float=1):
    """Function to generate a piecewise linear function with two segments
    Logic: suppose we have two line segments:
    line0 = slope0 * x + int0
    line1 = slope1 * x + int1
    Then we have:
    * int0: intercept of first segment when x == 0
    * int1: intercept of second segment when x == 0
    * (int1 - int0) is gap between intercepts at x == 0
    * chg_pt = (int1 - int0)/slope0 is the x-coordinate of line0 where line0(x) = int1
    * if we shift line1 by chg_pt then line1(x) = line0(x) at x == chg_pt since
        line1 = slope1 * (x - chg_pt) + int1 = int1 when x == chg_pt.
    * so at x=chg_pt, line0(x) = line1(x) = int1
    We can then left the final piecewise linear function be:
    * line0 when x < chg_pt
    * line1 when x > chg_pt
    * line0 = line1 when x == chg_pt
    
    Args:
        x (np.ndarray): input values
        slope0 (float): slope of first segment line0
        int0 (float): intercept of first segment line0 when x == 0
        slope1 (float): slope of second segment line1
        int1 (float): intercept of second segment line1 when x == 0 and no shift to line1
        scaling (float): scaling factor for piecewise linear function. If scaling < 0,
            then the piecewise linear function is flipped.
        
    
    Returns:
        np.ndarray: piecewise linear function values
    """        
    if slope0 != 0:
        chg_pt = (int1 - int0)/slope0
    else:
        # in case slope0 is zero, we set chg_pt to be very large
        chg_pt = (int1 - int0)/1e-6
    
    line0 = slope0 * x + int0
    line1 = slope1 * (x - chg_pt) + int1
    
    logger.debug(f"chg_pt: {chg_pt}")
    return scaling*((x <= chg_pt) * line0 + (x > chg_pt) * line1)
    

def shortest_path_synthetic_sym_no_noise(num_data: int, 
        num_features: int, 
        grid: Tuple[int, int], 
        deg: int=1, 
        rnd: np.random.RandomState=None,
        seed: int=135):
    """Function to generate the synthetic grid patterned shortest path experiment originally from
    Smart “Predict, then Optimize” paper (https://arxiv.org/pdf/1710.08005)
    and also used in:
    * PyEPO (https://arxiv.org/pdf/2206.14234)
    * Decision-Focused Learning with Directional Gradients (https://arxiv.org/pdf/2402.03256)
    
    The grid experiment has:
    * underlying grid network that is grid[0]-by-grid[1] where each element of grid is node
    * edges go only from left to right (west to east) and from top to bottom (north to south)
    * goal is to find the shortest path from the top-left node to the bottom-right node
    
    The synthetic data generation has:
    1. feature vectors of shape (num_data, num_features) and each feature is drawn from N(0, 1)
    2. cost vectors (each element of cost vector corresponds to an edge in the grid network) of shape (num_data, num_edges)
        - cost of each edge is generated as a polynomial of the dot product of the feature vector and a random matrix B
          (see original paper section 6.1 https://arxiv.org/pdf/1710.08005)
    
    Args:
        num_data (int): number of data samples in experiment
        num_features (int): dimension of features
        grid (Tuple[int, int]): specifies grid network size is grid[0]-by-grid[1] many nodes        
        deg (int): deg specifies the degree of the polynomial used in the 
            polynomial of the dot product of feature vector and random matrix B
            used to generate the cost vector. When using linear prediction model,
            higher deg corresponds to more misspecification of prediction model.
        seed (int): random seed for reproducibility

    Returns:
       dict: dictionary containing feature vectors, true cost vectors, and noisy cost vectors
    """
    # set random seed
    if rnd is None:
        rnd = np.random.RandomState(seed)

    # deg should be positive integer
    if type(deg) is not int:
        raise ValueError("deg = {} should be int.".format(deg))
    if deg <= 0:
        raise ValueError("deg = {} should be positive.".format(deg))
    
    """To instantiate instance of shortest path grid problem, we need to specify:
    * n = number of data points
    * p = dimension of features
    * d = dimension of cost vector/number of edges in the grid
    * B = random matrix parameter of bernoulli(0.5) random variables where B[i,j]
        toggles if feature i is used in generating cost of edge j"""
    n = num_data # number of data points
    p = num_features # dimension of features
    d = (grid[0] - 1) * grid[1] + (grid[1] - 1) * grid[0] # calculates number of edges in grid, which is also dimension of the cost vector    
    rnd_B = np.random.RandomState(1)
    B = rnd_B.binomial(1, 0.5, (d, p)) # random matrix of bernoulli(0.5) B of shape [d (number of edges), p (number of features)]     
    x = rnd.normal(0, 1, (n, p)) # feature vectors of shape [n (number of samples), p (feature dimension)] drawn from N(0, 1)
    x_plant = rnd.uniform(0, 2, (n, 1)) # feature value used for computing good and bad path edge costs
    
    """Generate (no noise) cost vectors c of shape [n (number of samples), d (number of edges)] 
    where c[i,j] is cost of edge j for experiment sample i. 
    
    c is generated by implementing a polynomial cost function where we:
    1. compute the dot product of feature vector x[i] and random matrix B 
        - x (n, p) @ B.T (p, d) -> c (n, d), where c[i,j] is cost of edge j for experiment sample i and 
          c[i,j] = x[i,:] * B.T [:,j] (dot product of feature vector of sample i and random matrix B column j)
    2. pass the dot product through a polynomial kernel of degree deg
    3. rescale the cost vector by dividing by 3.5^deg (rescaling implemented in PyEPO, not in original Smart “Predict, then Optimize” paper)
    """
    c = x @ B.T
    c = ((c / np.sqrt(p)) + 3)**deg + 1 # polynomial kernel of x dot product B with degree deg
    c = c/(3.5 ** deg) # rescaling implemented in PyEPO (not in original Smart “Predict, then Optimize” paper)
    
    data = {'x': x, 'x_plant': x_plant, 'c': c}
    return data


def shortest_path_synthetic_plant_path(x_plant: np.ndarray, 
                                    c: np.ndarray, 
                                    planted_good_edges: list=None, 
                                    planted_bad_edges: list=None,
                                    planted_good_pwl_params: dict=None,
                                    planted_bad_pwl_params: dict=None):
    """
    Modifies the cost vectors c by planting good and bad paths in the cost vectors.
    
    Args: 
        x (np.ndarray): feature vectors of shape (num_data, num_features)
        c (np.ndarray): cost vectors of shape (num_data, num_edges)
        planted_good_edges (list): list of good edges to plant
        planted_bad_edges (list): list of bad edges to plant
        planted_good_pwl_params (dict): dictionary of parameters {'slope0', 'int0', 'slope1', 'int1'} for good edges input to piecewise_linear
        planted_bad_pwl_params (dict): dictionary of parameters {'slope0', 'int0', 'slope1', 'int1'} for bad edges input to piecewise_linear
        
    Returns:
        dict: dictionary containing planted feature vectors and cost vectors
    """
    
    """
    check if planted_good_edges and planted_bad_edges are properly passed in,
    if not, set them to default values and instantiate necessary variables.
    Values are set based on https://arxiv.org/pdf/2402.03256 response letter where good and
    bad paths are planted with specific cost instantiations
    """    
    if planted_good_edges is None:
        planted_good_edges = np.array([1, 4, 9, 16, 24, 31, 36, 39])
    if planted_bad_edges is None:
        planted_bad_edges = np.array([0, 3, 8, 15, 23, 30, 35, 38])
    good_bad_edges = np.concatenate((planted_good_edges, planted_bad_edges)) # combined good and bad edges
    
    # remaining edges that are not good or bad
    remain_edges = np.setdiff1d(np.arange(c.shape[1]), # all possible edges
                                good_bad_edges)
    logger.debug(f"good_bad_edges: {good_bad_edges}, remain_edges: {remain_edges}")
    
    if planted_good_pwl_params is None:
        planted_good_pwl_params = {'slope0': 0, 'int0': 2.2, 'slope1': 0, 'int1': 2.2}
    if planted_bad_pwl_params is None:
        planted_bad_pwl_params = {'slope0': 4, 'int0': 0, 'slope1': 0, 'int1': 2.2}
            
    
    # bad paths
    c = c.copy()
    c[:, planted_good_edges] = piecewise_linear(x=x_plant,
                                            slope0=planted_good_pwl_params['slope0'], 
                                            int0=planted_good_pwl_params['int0'],
                                            slope1=planted_good_pwl_params['slope1'],
                                            int1=planted_good_pwl_params['int1'])
    # good path
    c[:, planted_bad_edges] = piecewise_linear(x=x_plant,
                                            slope0=planted_bad_pwl_params['slope0'], 
                                            int0=planted_bad_pwl_params['int0'],
                                            slope1=planted_bad_pwl_params['slope1'],
                                            int1=planted_bad_pwl_params['int1'])
    c[:, remain_edges] += 2.2
    
    # planted data
    data = {'x_plant': x_plant, 'c_plant': c}
    return data    


def add_noise(c: np.ndarray,
        noise_type: str='unif',
        noise_width: float=0, 
        rnd: np.random.RandomState=None,
        seed: int=135):
    """Function to add noise to cost vectors c

    Args:
        c (np.ndarray): costs to add noise to
        noise_type (str): type of noise added to cost vector. Defaults to 'unif'.
        noise_width (float): degree of noise added to cost vector. Defaults to 0.
        rnd (np.random.RandomState): random state object for reproducibility 
        seed (int): random seed for reproducibility. Defaults to 135.

    Returns:
        dict: dictionary containing noisy cost vectors and noise vectors
    """
    # set random seed
    if rnd is None:
        rnd = np.random.RandomState(seed)
        
    # instantiate noise functions
    # epsilon_gen_func contains the noise generation functions for uniform and normal noise
    epsilon_gen_func = {'unif': lambda x: rnd.uniform(low=1 - noise_width, 
                                                    high=1 + noise_width, 
                                                    size=x.shape),
                        'normal': lambda x: rnd.normal(loc=0, 
                                                     scale=noise_width, 
                                                     size=x.shape)}
    # as implemented in https://arxiv.org/pdf/2402.03256,
    # when noise_type is 'unif', noise is generated as a uniform random variable and multiplied to cost vector
    # when noise_type is 'normal', noise is generated as a normal random variable and added to cost vector
    noise_func = {'unif': lambda x, y: x * y,
                  'normal': lambda x, y: x + y}
    
    # add noise to cost vectors
    epsilon = epsilon_gen_func[noise_type](c)
    c_hat = noise_func[noise_type](c, epsilon)
    
    data = {'cost_noise': c_hat, 'noise': epsilon}
    return data


def genData(num_data: int, 
        num_features: int, 
        grid: Tuple[int, int], 
        deg: int=1, 
        noise_type: str='unif',
        noise_width: float=0, 
        seed: int=135,
        plant_edges: bool=True,
        planted_good_edges: list=None, 
        planted_bad_edges: list=None,
        planted_good_pwl_params: dict=None,
        planted_bad_pwl_params: dict=None):
    """Orchestration function to generate synthetic shortest path data:
    1. Generate synthetic shortest path data with no noise - costs and features
    2. If plant_edges == True, plant good and bad paths in the cost vectors
    3. Add noise to cost vectors

    Args:
        num_data (int): number of data samples in experiment
        num_features (int): dimension of features
        grid (Tuple[int, int]): specifies grid network size is grid[0]-by-grid[1] many nodes        
        deg (int): deg specifies the degree of the polynomial used in the 
            polynomial of the dot product of feature vector and random matrix B
            used to generate the cost vector. When using linear prediction model,
            higher deg corresponds to more misspecification of prediction model.
        noise_type (str): type of noise added to cost vector. Defaults to 'unif'.
        noise_width (float): degree of noise added to cost vector. Defaults to 0.
        seed (int): random seed for reproducibility. Defaults to 135.        
        plant_edges (bool, optional): to plant edges or not. Defaults to True.
        planted_good_edges (list): list of good edges to plant
        planted_bad_edges (list): list of bad edges to plant
        planted_good_pwl_params (dict): dictionary of parameters {'slope0', 'int0', 'slope1', 'int1'} for good edges input to piecewise_linear
        planted_bad_pwl_params (dict): dictionary of parameters {'slope0', 'int0', 'slope1', 'int1'} for bad edges input to piecewise_linear
    Returns:
        dict: dictionary containing feature vectors, true cost vectors, noisy cost vectors, and noise vectors
    """
    rnd = np.random.RandomState(seed) 
    
    # 1. contents of data =  {'x': x, 'c': c}
    data = shortest_path_synthetic_sym_no_noise(num_data=num_data,
        num_features=num_features, 
        grid=grid, 
        deg=deg, 
        rnd=rnd)
    
    # 2. plant edges
    if plant_edges:
        # contents of data_plant =  {'x_plant': x_plant, 'c_plant': c}    
        data_plant = shortest_path_synthetic_plant_path(x_plant=data['x_plant'],
                                        c=data['c'],
                                        planted_good_edges=planted_good_edges, 
                                        planted_bad_edges=planted_bad_edges,
                                        planted_good_pwl_params=planted_good_pwl_params,
                                        planted_bad_pwl_params=planted_bad_pwl_params)
        data['c_plant'] = data_plant['c_plant']
        data['x'] = np.concatenate((data['x'], data_plant['x_plant']), axis = 1) # combine feature data
        
    else: 
        # if don't plant edges, set c_plant to c so we have common keys between both cases where plant_edges == True
        # and plant_edges == False. This way easier to maintain code, have less if else statements branching
        data['c_plant'] = data['c']
        
    # 3. add noise to cost vectors - contents of data_noise = {'cost_noise': c_hat, 'noise': epsilon}   
    data_noise = add_noise(c=data['c_plant'],
            noise_type=noise_type,
            noise_width=noise_width,
            rnd=rnd)
    
    # aggregate data    
    final_data = {'feat': data['x'],
                  'cost_noplant': data['c'], # cost vectors with no planted good and bad paths, no noise
                  'cost_true': data['c_plant'], # cost vectors with planted good and bad paths, no noise
                  'cost': data_noise['cost_noise'], # cost vectors with planted good and bad paths, with noise
                  'epsilon': data_noise['noise']}
    return final_data                  



