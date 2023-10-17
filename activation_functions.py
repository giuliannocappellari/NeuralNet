import numpy as np
from abc import ABC, abstractclassmethod


class ActivationFunc(ABC):
   
  @abstractclassmethod
  def func():
      ... 
    
  @abstractclassmethod
  def grad():
      ... 
    
  @abstractclassmethod
  def get_delta():
      ... 


class LinearActivation(ActivationFunc):
    
    @staticmethod
    def func(s:np.array) -> float:
    
        """
        This function compute linerar activation

        Parameters
        ----------
            s:np.array
                The input array to pass through activation function

        Return
        ----------
            o:np.array
                The numpy array passed through linear activation function
        """
    
        o = s
        return o

    @staticmethod
    def grad(s:np.array) -> np.array:

        """
        This function compute linerar gradient

        Parameters
        ----------
            s:np.array
                The input array to get gradient

        Return
        ----------
            np.array
                The numpy array gotten from linear activation gradient
        """

        return np.ones_like(s)

    @staticmethod
    def get_delta(dJ_ds:np.array, d_o_s:np.array):

        """
        This function compute delta linear gradient

        Parameters
        ----------
            dJ_ds:np.array
                The output-coust gradient 
            d_o_s:np.array
                The output-activation gradient 

        Return
        ----------
            np.array
                The numpy array gotten from linear activation gradient
        """

        return dJ_ds * d_o_s


class Sigmoid(ActivationFunc):

    @staticmethod
    def func(s:np.array) -> float:
    
        """
        This function compute sigmoid activation

        Parameters
        ----------
            s:np.array
                The input array to pass through activation function

        Return
        ----------
            o:np.array
                The numpy array passed through sigmoid activation function
        """

        return 1 / (1 + np.exp(-s))

    @staticmethod
    def grad(s:np.array) -> np.array:

        """
        This function compute sigmoid gradient

        Parameters
        ----------
            s:np.array
                The input array to get gradient

        Return
        ----------
            np.array
                The numpy array gotten from sigmoid activation gradient
        """

        return s * (1 - s)

    @staticmethod
    def get_delta(d_J_o:np.array, d_o_s:np.array):

        """
        This function compute delta sigmoid gradient

        Parameters
        ----------
            dJ_ds:np.array
                The output-coust gradient 
            d_o_s:np.array
                The output-activation gradient 

        Return
        ----------
            np.array
                The numpy array gotten from sigmoid activation gradient
        """

        return d_J_o * d_o_s


class Relu(ActivationFunc):

    @staticmethod
    def func(s:np.array) -> np.array:

        """
        This function compute relu activation

        Parameters
        ----------
            s:np.array
                The input array to pass through activation function

        Return
        ----------
            o:np.array
                The numpy array passed through relu activation function
        """

        return np.maximum(0,s)
  
    @staticmethod
    def grad(s:np.array) -> np.array:

        """
        This function compute relu gradient

        Parameters
        ----------
            s:np.array
                The input array to get gradient

        Return
        ----------
            np.array
                The numpy array gotten from relu activation gradient
        """

        return np.greater(s, 0).astype(int)
  
    @staticmethod
    def get_delta(d_J_o:np.array, d_o_s:np.array):

        """
        This function compute delta relu gradient

        Parameters
        ----------
            dJ_ds:np.array
                The output-coust gradient 
            d_o_s:np.array
                The output-activation gradient 

        Return
        ----------
            np.array
                The numpy array gotten from relu activation gradient
        """        

        return d_J_o * d_o_s


class Tanh(ActivationFunc):

    @staticmethod
    def func(s:np.array) -> np.array:
        """
        This function computes the tanh activation.

        Parameters
        ----------
            s:np.array
                The input array to pass through activation function.

        Return
        ----------
            o:np.array
                The numpy array passed through the tanh activation function.
        """
        return np.tanh(s)

    @staticmethod
    def grad(s:np.array) -> np.array:
        """
        This function computes the gradient of the tanh activation.

        Parameters
        ----------
            s:np.array
                The input array to get gradient.

        Return
        ----------
            np.array
                The numpy array derived from the gradient of the tanh activation function.
        """
        return 1.0 - np.square(np.tanh(s))

    @staticmethod
    def get_delta(d_J_o:np.array, d_o_s:np.array):
        """
        This function computes the delta for the gradient of the tanh activation.

        Parameters
        ----------
            dJ_ds:np.array
                The output-cost gradient.
            d_o_s:np.array
                The output-activation gradient.

        Return
        ----------
            np.array
                The numpy array derived from the gradient of the tanh activation function.
        """
        return d_J_o * d_o_s



class Softmax(ActivationFunc):

    @staticmethod
    def func(x:np.array) -> np.array:
        """
        This function computes the Softmax activation.

        Parameters
        ----------
            x:np.array
                The input array for which to compute the Softmax.

        Return
        ----------
            o:np.array
                The numpy array after applying the Softmax function.
        """
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    @staticmethod
    def grad(s:np.array) -> np.array:
        """
        This function computes the gradient of the Softmax activation.
        Note: In practice, this gradient is rarely used directly since
              Softmax is often paired with a categorical cross-entropy loss,
              simplifying the backpropagation process.

        Parameters
        ----------
            s:np.array
                The input array to get gradient.

        Return
        ----------
            np.array
                The numpy array derived from the gradient of the Softmax activation.
        """
        p = Softmax.func(s)
        return p * (1 - p)

    @staticmethod
    def get_delta(d_J_o:np.array, d_o_s:np.array) -> np.array:
        """
        This function computes the delta for the gradient of the Softmax activation.

        Parameters
        ----------
            dJ_ds:np.array
                The output-cost gradient.
            d_o_s:np.array
                The output-activation gradient.

        Return
        ----------
            np.array
                The numpy array derived from the gradient of the Softmax activation.
        """
        return np.sum(d_J_o * d_o_s, keepdims=True, axis=1)

  

class LeakyRelu(ActivationFunc):

    @staticmethod
    def func(s:np.array) -> np.array:
        """
        This function computes the Leaky ReLU activation.

        Parameters
        ----------
            s:np.array
                The input array to pass through activation function.

        Return
        ----------
            o:np.array
                The numpy array passed through the Leaky ReLU activation function.
        """
        return np.where(s > 0, s, s * 0.01)

    @staticmethod
    def grad(s:np.array, alpha=0.01) -> np.array:
        """
        This function computes the gradient of the Leaky ReLU activation.

        Parameters
        ----------
            s:np.array
                The input array to get gradient.
            alpha:float, optional
                The slope coefficient for negative values, default is 0.01.

        Return
        ----------
            np.array
                The numpy array derived from the gradient of the Leaky ReLU activation function.
        """
        dx = np.ones_like(s)
        dx[s < 0] = alpha
        return dx

    @staticmethod
    def get_delta(d_J_o:np.array, d_o_s:np.array):
        """
        This function computes the delta for the gradient of the Leaky ReLU activation.

        Parameters
        ----------
            dJ_ds:np.array
                The output-cost gradient.
            d_o_s:np.array
                The output-activation gradient.

        Return
        ----------
            np.array
                The numpy array derived from the gradient of the Leaky ReLU activation function.
        """
        return d_J_o * d_o_s

  