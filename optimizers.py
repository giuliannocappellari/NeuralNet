import numpy as np
from typing import Tuple
from abc import ABC, abstractclassmethod, abstractmethod
from layers import *


class Optimizer(ABC):

    def __init__(self, lr=0.01) -> None:
        self.lr = lr

    @abstractmethod
    def update(self, t=0, weight=0, bias=0, dw=0, db=0):
        pass


class GradientDescendent(Optimizer):

    def __init__(self, lr=0.01) -> None:
        """
            Initialize GradientDescendent Optimizer

            Parameters
            ----------
                lr : float
                    The learning rate

            Examples
            --------
                gd = GradientDescendent()
        """
        super().__init__(lr)


    def update(self, layer:NeuronMetaClass, t:int=1) -> Tuple[np.array, np.array]:

        """
            Update the weights using Gradient Descendent

            Parameters
            ----------
                weight : np.array
                    The weights array
                bias : np.array
                    The bias array
                dw: np.array
                    The weights gradient vector array 
                db: np.array
                    The bias gradient vector array 

            Returns
            -------
                (weight, bias)
                    A tuple containing the new weights and bias
            Examples
            --------
                weight, bias = gd.update(weight=weight, bias=bias, dw=dw, db=db)
        """
        layer.weights = layer.weights - (self.lr * layer.d_J)
        layer.bias = layer.bias - (self.lr * layer.d_B)


class Adam(Optimizer):

    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8) -> None:

        """
            Initialize AdamOptim

            Parameters
            ----------
                lr : float
                    The learning rate
                beta1 : float
                    A momentum control variable
                beta2 : float
                    A velocity control variable
                epsilon: float
                    A variable to avoid division per zero

            Examples
            --------
                adam = Adam()
        """

        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon


    def update(self, layer:NeuronMetaClass, t:int=1) -> Tuple[np.array, np.array]:

        """
            Update the weights using adam

            Parameters
            ----------
                t : int
                    The epoch number
                weight : np.array
                    The weights array
                bias : np.array
                    The bias array
                dw: np.array
                    The weights gradient vector array 
                db: np.array
                    The bias gradient vector array 

            Returns
            -------
                (weight, bias)
                    A tuple containing the new weights and bias
            Examples
            --------
                weight, bias = adam.update(epoch, weight=weight, bias=bias, dw=dw, db=db)
        """
        layer.momentum_weights = self.beta1*layer.momentum_weights + ((1-self.beta1)*layer.d_J)
        layer.momentum_bias = self.beta1*layer.momentum_bias + ((1-self.beta1)*layer.d_B)
        layer.velocity_weights = self.beta2*layer.velocity_weights + ((1-self.beta2)*(layer.d_J**2))
        layer.velocity_bias = self.beta2*layer.velocity_bias + ((1-self.beta2)*(layer.d_B**2))
        
        momentum_weights_corr = layer.momentum_weights/(1-self.beta1**t)
        momentum_bias_corr = layer.momentum_bias/(1-self.beta1**t)
        velocity_weights_corr = layer.velocity_weights/(1-self.beta2**t)
        velocity_bias_corr = layer.velocity_bias/(1-self.beta2**t)
        
        layer.weights = layer.weights - self.lr*(momentum_weights_corr/(np.sqrt(velocity_weights_corr)+self.epsilon))
        layer.bias = layer.bias - self.lr*(momentum_bias_corr/(np.sqrt(velocity_bias_corr)+self.epsilon))
        


if __name__ == "__main__":
    adam = Adam()