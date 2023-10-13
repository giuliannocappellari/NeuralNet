import numpy as np
from typing import Callable
from activation_functions import Sigmoid


def initialize_weights(neurons:int, input_size:int, activation__func:Callable) -> np.array:
    return np.array([np.random.normal(0, 2/neurons, input_size) for i in range(neurons)])

def he_normal(neurons:int, input_size:int) -> np.array:
    stddev = np.sqrt(2. / neurons)
    return np.random.normal(0, stddev, (neurons, input_size))

def he_uniform(neurons:int, input_size:int) -> np.array:
    limit = np.sqrt(6. / neurons)
    return np.random.uniform(-limit, limit, (neurons, input_size))

def xavier_normal(neurons:int, input_size:int) -> np.array:
    stddev = np.sqrt(2. / (neurons + input_size))
    return np.random.normal(0, stddev, (neurons, input_size))

def xavier_uniform(neurons:int, input_size:int) -> np.array:
    limit = np.sqrt(6. / (neurons + input_size))
    return np.random.uniform(-limit, limit, (neurons, input_size))

def normal(neurons:int, input_size:int) -> np.array:
    return np.random.normal(0, 1, (neurons, input_size))

def uniform(neurons:int, input_size:int) -> np.array:
    return np.random.uniform(-1, 1, (neurons, input_size))

def initialize_bias() -> np.array:
    return np.array(0)