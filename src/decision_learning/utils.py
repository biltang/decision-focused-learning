import inspect
import os 
import time
from functools import wraps

import logging 
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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


def log_runtime(func):
    """Decorator to log the runtime of a function"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Function '{func.__name__}' took {end_time - start_time} seconds to run.")
        return result
    return wrapper