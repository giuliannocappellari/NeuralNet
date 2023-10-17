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
    def func(y_hat:np.array, y:np.array) -> float:
        """
        This function computes the Mean Squared Error (MSE) loss.

        Parameters
        ----------
            y_hat:np.array
                The predicted values.
            y:np.array
                The true values or labels.

        Return
        ----------
            float
                The computed MSE loss value.
        """
        return np.mean((y - y_hat)**2)

    @staticmethod
    def grad(y_hat:np.array, y:np.array) -> np.array:
        """
        This function computes the gradient of the Mean Squared Error (MSE) with respect to the predictions.

        Parameters
        ----------
            y_hat:np.array
                The predicted values.
            y:np.array
                The true values or labels.

        Return
        ----------
            np.array
                The gradient of the MSE loss with respect to the predictions.
        """
        return y_hat - y

    

class BinaryCrossEntropy(LossFunc):

    @staticmethod
    def func(y_hat:np.array, y:np.array) -> float:
        """
        This function computes the Binary Cross-Entropy loss.

        Parameters
        ----------
            y_hat:np.array
                The predicted probability values (should be between 0 and 1).
            y:np.array
                The true binary labels (either 0 or 1).

        Return
        ----------
            float
                The computed Binary Cross-Entropy loss value.

        Notes
        ----------
            It's important for the `y_hat` values to be clipped or adjusted to avoid
            taking the logarithm of 0 or 1, which could lead to numerical instability.
        """
        y_hat = np.array([y_hat])
        y = np.array([y])
        return -np.mean(((1-y) * np.log(1-y_hat)) + (y * np.log(y_hat)), axis=0)

    @staticmethod
    def grad(y_hat:np.array, y:np.array) -> np.array:
        """
        This function computes the gradient of the Binary Cross-Entropy loss with respect to the predictions.

        Parameters
        ----------
            y_hat:np.array
                The predicted probability values (should be between 0 and 1).
            y:np.array
                The true binary labels (either 0 or 1).

        Return
        ----------
            np.array
                The gradient of the Binary Cross-Entropy loss with respect to the predictions.

        Notes
        ----------
            It's important for the gradient computation to consider potential division by zero 
            if `y_hat` or `(1 - y_hat)` is too close to zero.
        """
        return (y_hat - y) / (y_hat * (1 - y_hat))



class CrossEntropyLoss(LossFunc):
    
    @staticmethod
    def func(y_true: np.array, y_pred: np.array) -> float:
        """
        This function computes the Cross-Entropy loss for multi-class classification.

        Parameters
        ----------
            y_true: np.array
                The true one-hot encoded class labels.
            y_pred: np.array
                The predicted probability values for each class (should sum to 1 for each instance).

        Return
        ----------
            float
                The computed Cross-Entropy loss value.

        Notes
        ----------
            To prevent numerical instability, the predictions are clipped to be in the range 
            [epsilon, 1-epsilon] to avoid taking the logarithm of 0 or 1.
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return -np.sum(y_true * np.log(y_pred))

    @staticmethod
    def grad(y_true: np.array, y_pred: np.array) -> np.array:
        """
        This function computes the gradient of the Cross-Entropy loss with respect to the predictions.

        Parameters
        ----------
            y_true: np.array
                The true one-hot encoded class labels.
            y_pred: np.array
                The predicted probability values for each class.

        Return
        ----------
            np.array
                The gradient of the Cross-Entropy loss with respect to the predictions.
                
        Notes
        ----------
            The returned gradient will have a shape that's transposed relative to the input predictions.
        """
        return y_pred.T - y_true


