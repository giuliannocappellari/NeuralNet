import numpy as np
from typing import Callable, Tuple
from initialize import *
from activation_functions import Sigmoid, ActivationFunc
from abc import ABC, abstractclassmethod


class LayerMetaClass(ABC):

    def __init__(self, input_size:int, neurons:int, activation_func:ActivationFunc=Sigmoid, seed:int=1234,
                 initializer_weights:WeightsInitialize=HeNormal, initializer_bias:InitializeBias=InitializeBias) -> None:
        
        """
        Initialize the neuron meta class

        Parameters
            ----------
                input_size:int
                    The layers input size
                neurons:int
                    The number of neurons in layer
                activation_func:ActivationFunc
                    The activation function of the layer
                seed:int
                initialize_weights:WeightsInitialize
                    The weight's initialization
                initialize_bias:InitializeBias   
                    The bias's initialization
                
        """
        
        self.input_size = input_size
        self.neurons = neurons
        self.activation_func = activation_func
        self.weights = initializer_weights(neurons, input_size, seed).initialize_weights()
        self.delta = 0 
        self.d_W = 0
        self.d_B = 0
        self.bias = initializer_bias(neurons).initialize_bias()
        self.momentum_weights = np.zeros_like(self.neurons)
        self.velocity_weights = np.zeros_like(self.neurons)
        self.momentum_bias = np.zeros_like(self.bias)
        self.velocity_bias = np.zeros_like(self.bias)

    @abstractclassmethod
    def zero_grad(self, batch_size) -> Tuple[np.array, np.array, np.array]:
        
        """
        To zero out the layer's delta, and gradients 

        Parameters
        ----------
            batch_size:int
                The number of rows passed from dataset to layer (should be your batch_size) 

        Return
        ----------
            delta:np.array
                Zero numpy array with shape (neuros, batch_size) 
            d_J:np.array
                The weights gradient, Zero numpy array with shape (self.neurons, self.input_size) 
            d_B:np.array
                The bias gradient, Zero numpy array with shape (1,1) 
        """

        ...
    
    @abstractclassmethod
    def forward(self, x: np.array) -> np.array:

        """
        Compute the layer's forward

        Parameters
        ----------
            x:np.array
                The input array to pass through layer

        Return
        ----------
            o:np.array
                The numpy array containing layer output with shape(batch_size x neurons)
        """

        ...
    

    @abstractclassmethod
    def backward(self, dJ_ds) -> np.array:

        """
        Compute the layer's backward

        Parameters
        ----------
            x:np.array
                The input array to pass through layer

        Return
        ----------
            o:np.array
                The numpy array containing layer output with shape(batch_size x neurons)
        """

        ...


class Linear(LayerMetaClass):

    def __init__(self, input_size:int, neurons:int, activation_func:ActivationFunc=Sigmoid, seed:int=1234,
                 initialize_weights:WeightsInitialize=HeNormal, initialize_bias:InitializeBias=InitializeBias) -> None:
        super().__init__(input_size, neurons, activation_func, seed,initialize_weights, initialize_bias)


    def zero_grad(self, batch_size) -> Tuple[np.array, np.array, np.array]:
        
        """
        To zero out the layer's delta, and gradients 

        Parameters
        ----------
            batch_size:int
                The number of rows passed from dataset to layer (should be your batch_size) 

        Return
        ----------
            delta:np.array
                Zero numpy array with shape (neuros, batch_size) 
            d_J:np.array
                The weights gradient, Zero numpy array with shape (self.neurons, self.input_size) 
            d_B:np.array
                The bias gradient, Zero numpy array with shape (1,1) 
        """
        
        delta = np.zeros((self.neurons, batch_size))
        d_W = np.zeros((self.neurons, self.input_size))
        d_B = np.zeros((1, 1))
        return delta, d_W, d_B
    

    def forward(self, x: np.array) -> np.array:

        """
        Compute the layer's forward

        Parameters
        ----------
            x:np.array
                The input array to pass through layer

        Return
        ----------
            o:np.array
                The numpy array containing layer output with shape (batch_size x neurons)
        """
        s = np.dot(x, self.weights.T) + self.bias.T
        o = self.activation_func.func(s)
        self.input = x
        self.s = s
        self.o = o
        return o
    

    def backward(self, dJ_ds) -> np.array:

        """
        Compute the layer's backward

        Parameters
        ----------
            x:np.array
                The input array to pass through layer

        Return
        ----------
            o:np.array
                The numpy array containing layer output with shape (batch_size x neurons)
        """

        do_ds = self.activation_func.grad(self.s)
        self.delta += self.activation_func.get_delta(dJ_ds, do_ds)
        self.d_W += np.dot(self.input.T, self.delta) 
        self.d_B += np.sum(self.delta, axis=0).reshape(-1,1)
        dJ_ds = np.dot(self.delta, self.weights) 
        
        return dJ_ds


if __name__ == "__main__":
    nn = Linear(3, 1, Sigmoid(), initialize_weights=XavierNormal)
    print(nn.weights)