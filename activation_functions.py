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
    return s

  @staticmethod
  def grad(s:np.array) -> np.array:
    return np.ones_like(s)

  @staticmethod
  def get_delta(d_J_o:np.array, d_o_s:np.array):
    return d_J_o * d_o_s


class Sigmoid(ActivationFunc):

  @staticmethod
  def func(s:np.array) -> float:
    
    """
    This function gets the s (out of the last layer * weights) parameter
    And pass it for relu function
    """
    return 1 / (1 + np.exp(-s))

  @staticmethod
  def grad(s:np.array) -> np.array:
    return s * (1 - s)

  @staticmethod
  def get_delta(d_J_o:np.array, d_o_s:np.array):
    return d_J_o * d_o_s


class Relu(ActivationFunc):

  @staticmethod
  def func(s:np.array) -> np.array:
    return np.maximum(0,s)
  
  @staticmethod
  def grad(s:np.array) -> np.array:
    return np.greater(s, 0).astype(int)
  
  @staticmethod
  def get_delta(d_J_o:np.array, d_o_s:np.array):
    return d_J_o * d_o_s


class Tanh(ActivationFunc):

  @staticmethod
  def func(s:np.array) -> np.array:
    return np.tanh(s)

  @staticmethod
  def grad(s:np.array) -> np.array:
    return 1.-np.tanh(s)**2

  @staticmethod
  def get_delta(d_J_o:np.array, d_o_s:np.array):
    return d_J_o * d_o_s


class Softmax(ActivationFunc):

  @staticmethod
  def func(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

  @staticmethod
  def grad(s):
    p = Softmax.func(s)
    return p * (1 - p)
  
  @staticmethod
  def get_delta(d_J_o:np.array, d_o_s:np.array):
    return np.sum(d_J_o * d_o_s, keepdims=True, axis=1)
  

class LeakyRelu(ActivationFunc):
   
  @staticmethod
  def func(s:np.array) -> np.array:
    return np.where(s > 0, s, s * 0.01)  

  @staticmethod
  def grad(s:np.array, alpha=0.01) -> np.array:
    dx = np.ones_like(s)
    dx[s < 0] = alpha
    return dx
  
  @staticmethod
  def get_delta(d_J_o:np.array, d_o_s:np.array):
    return d_J_o * d_o_s
  