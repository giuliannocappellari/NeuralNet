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

  def grad(s:np.array) -> np.array:
    return np.greater(s, 0).astype(int)
  
  #TODO
  @staticmethod
  def get_delta():
     pass


class Tanh(ActivationFunc):

  @staticmethod
  def tanh(s:np.array) -> np.array:
    return np.tanh(s)

  @staticmethod
  def tanh_grad(s:np.array) -> np.array:
    return 1.-np.tanh(s)**2

  #TODO
  @staticmethod
  def get_delta():
     pass


class Softmax(ActivationFunc):

  @classmethod
  def softmax(x):
      e_x = np.exp(x - np.max(x))
      return e_x / e_x.sum(axis=0)

  @classmethod
  def softmax_grad(s): 
      jacobian_m = np.diag(s)
      for i in range(len(jacobian_m)):
          for j in range(len(jacobian_m)):
              if i == j:
                  jacobian_m[i][j] = s[i] * (1-s[i])
              else: 
                  jacobian_m[i][j] = -s[i] * s[j]
      return jacobian_m
  
  def get_delta(d_J_o:np.array, d_o_s:np.array):
      return np.sum(d_J_o * d_o_s, keepdims=True, axis=1)
  

if __name__ == "__main__":
  af = Sigmoid()
  print(af.func(np.array([1,2,3])))