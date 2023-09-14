import numpy as np
from typing import Callable
from initialize import initialize_weights
from activation_functions import Sigmoid, ActivationFunc
from abc import ABC


class NeuronMetaClass(ABC):

    def __init__(self, input_size:int, neurons:int, activation_func:ActivationFunc=Sigmoid) -> None:
        
        self.input_size = input_size
        self.neurons = neurons
        self.activation_func = activation_func
        self.weights = initialize_weights(neurons, input_size, activation_func)
        self.bias = 0
        self.momentum_weights = 0
        self.velocity_weights = 0
        self.momentum_bias = 0
        self.velocity_bias = 0

    
    def zero_grad(self, input_len):
        self.delta = np.zeros((self.neurons, input_len))
        self.d_J = np.zeros((self.neurons, self.input_size))


class Linear(NeuronMetaClass):

    def __init__(self, input_size:int, neurons:int, activation_func:ActivationFunc) -> None:
        super().__init__(input_size, neurons, activation_func)


if __name__ == "__main__":
    nn = Linear(input_size=2, neurons=3, activation_func=Sigmoid)
    print(nn.weights)