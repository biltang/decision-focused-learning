from abc import ABC, abstractmethod

from torch import nn

# ------------------------------------------------------------------------------
# PREDICTION MODELS
# ------------------------------------------------------------------------------
class LinearRegression(nn.Module):

    def __init__(self, input_dim: int, output_dim: int):
        """Instantiate linear regression model

        Args:
            input_dim (int): number of input features
            output_dim (int): number of output features
        """
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out
    
    
class MLP(nn.Module):
    def __init__(self, 
                input_size: int, 
                hidden_sizes: list, 
                output_size: int, 
                activation_fn: callable=nn.ReLU, 
                dropout_p: float=0.0):
        """
        Args:
          input_size (int): Number of input features.
          hidden_sizes (list of int): List containing the number of units for each hidden layer.
          output_size (int): Number of output units.
          activation_fn (nn.Module): Activation function class to be used between layers (default: nn.ReLU).
          dropout_p (float): Dropout probability to be used after each hidden layer (default: 0.0).
        """
        super(MLP, self).__init__()
        
        layers = []
        in_size = input_size
        
        # Add hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(activation_fn())
            if dropout_p > 0.0:
                layers.append(nn.Dropout(dropout_p))
            in_size = hidden_size

        # Add output layer
        layers.append(nn.Linear(in_size, output_size))

        # Combine layers in a Sequential module
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
    
# ------------------------------------------------------------------------------
# OPTMIZATION MODELS
# ------------------------------------------------------------------------------
class OptModel(ABC):
    """Abstract base class for optimization models used as input into decision-focused loss functions to ensure
    that all optimization models have the same interface. They must implement the functions:
    - setObj: Set the optimization model's objective function.
    - solve: Solve the optimization model and return the optimal solution.    
    """
    @abstractmethod
    def __init__(self, modelSense: int=1):
        self.modelSense = modelSense
        
    
    @abstractmethod
    def setObj(self, x):
        raise NotImplementedError
    
    @abstractmethod
    def solve(self):
        raise NotImplementedError
    
    
def optmodel_interface():
    pass