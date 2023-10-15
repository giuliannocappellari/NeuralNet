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
        print("LOSS")
        print(y_hat)
        print(y)
        print(np.mean((y - y_hat)**2))
        print("LOSS")
        return np.mean((y - y_hat)**2)
    
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
        return (y_hat - y.T) / (y_hat * (1 - y_hat))


class CrossEntropyLoss:
    
    @staticmethod
    def func(y_true: np.array, y_pred: np.array) -> float:

        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return -np.sum(y_true * np.log(y_pred))

    @staticmethod
    def grad(y_true: np.array, y_pred: np.array) -> np.array:
        
        return y_pred.T - y_true

