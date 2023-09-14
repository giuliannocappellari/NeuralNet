import numpy as np
from abc import ABC, abstractclassmethod


class LossFunc(ABC):
   
    @abstractclassmethod
    def func():
        ... 
   
    @abstractclassmethod
    def grad():
        ...


class MSE(LossFunc):

    @staticmethod
    def func(y_hat:np.array, y:np.array):
        y_hat = np.array([y_hat])
        y = np.array([y])
        return (1/(2*y_hat.shape[0]))*sum([(y_hat[i] - y[i])**2 for i in range(y_hat.shape[0])])
    
    @staticmethod
    def grad(y_hat:np.array, y:np.array):
        return y_hat - y
    

class BinaryCrossEntropy(LossFunc):

    @staticmethod
    def func(y_hat:np.array, y:np.array):
        y_hat = np.array([y_hat])
        y = np.array([y])
        return -np.mean(((1-y) * np.log(1-y_hat)) + (y * np.log(y_hat)), axis=0)
    
    @staticmethod
    def grad(y_hat:np.array, y:np.array):
        return (y_hat - y) / (y_hat * (1 - y_hat))