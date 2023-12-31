import numpy as np
from typing import Callable
from initialize import *
from activation_functions import Sigmoid, ActivationFunc
from abc import ABC


class NeuronMetaClass(ABC):

    def __init__(self, input_size:int, neurons:int, activation_func:ActivationFunc=Sigmoid) -> None:
        
        self.input_size = input_size
        self.neurons = neurons
        self.activation_func = activation_func
        self.weights = initialize_weights(neurons, input_size, activation_func)



        self.bias = initialize_bias()
        self.momentum_weights = 0
        self.velocity_weights = 0
        self.momentum_bias = 0
        self.velocity_bias = 0

    
    def zero_grad(self, input_len):
        self.delta = np.zeros((self.neurons, input_len))
        self.d_J = np.zeros((self.neurons, self.input_size))
        self.d_B = np.zeros((1, 1))


class Linear(NeuronMetaClass):

    def __init__(self, input_size:int, neurons:int, activation_func:ActivationFunc, init_mode:str="default") -> None:
        super().__init__(input_size, neurons, activation_func)
        
        # if init_mode == "he_normal":
        #     self.weights = he_normal(neurons, input_size)
        # elif init_mode == "he_uniform":
        #     self.weights = he_uniform(neurons, input_size)
        # elif init_mode == "xavier_normal":
        #     self.weights = xavier_normal(neurons, input_size)
        # elif init_mode == "xavier_uniform":
        #     self.weights = xavier_uniform(neurons, input_size)
        # elif init_mode == "normal":
        #     self.weights = normal(neurons, input_size)
        # elif init_mode == "uniform":
        #     self.weights = uniform(neurons, input_size)


if __name__ == "__main__":
    nn = Linear(input_size=2, neurons=3, activation_func=Sigmoid)
    print(nn.weights)