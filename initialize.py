import numpy as np
from typing import Callable
from activation_functions import Sigmoid


def initialize_weights(neurons:int, input_size:int, activation__func:Callable) -> np.array:
    return np.array([np.random.normal(0, 2/neurons, input_size) for i in range(neurons)])