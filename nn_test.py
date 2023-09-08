import numpy as np
import math
from typing import List, Callable


def sigmoid(s) -> float:
  
  """
  This function gets the s (out of the last layer * weights) parameter
  And pass it for relu function
  """
  return 1 / (1 + np.exp(-s))


def deri_sigmoid(s:float) -> float:
  return s * (1 - s)


def MSE(y_hat:np.array, y:np.array):
    y_hat = np.array([y_hat])
    y = np.array([y])
    return (1/(2*y_hat.shape[0]))*sum([(y_hat[i] - y[i])**2 for i in range(y_hat.shape[0])])


class NN:


    def __init__(self, layers:List[dict], lr:float=0.5 ) -> None:
        
        self.layers = layers
        self.lr = lr
        self.loss = 0


    def forward(self, x:np.array) -> None:
        
        for layer in self.layers:
            s = np.dot(layer["weights"], x) + layer["bias"]
            x = layer["activation_func"](s)
            layer["out"] = x
    

    def compute_loss(self, y_hat:np.array, y:np.array, loss_func:Callable=MSE) -> None:

        self.loss = loss_func(y_hat, y)
    

    def backward(self, X, y) -> None:

        for i in range(len(self.layers)-1, -1, -1):
            if i == (len(self.layers) - 1):
                d_J_o = self.layers[i]["out"] - y
                d_o_s = self.layers[i]["out"] * (1 - self.layers[i]["out"])
                self.layers[i]["delta"] = d_J_o * d_o_s
                self.layers[i]["d_J"] = np.dot(self.layers[i]["delta"], self.layers[i-1]["out"].T)
            elif i == 0:
                d_o_s = self.layers[i]["out"] * (1 - self.layers[i]["out"])
                self.layers[i]["delta"] = np.dot(self.layers[i+1]["weights"].T, self.layers[i+1]["delta"]) * d_o_s
                self.layers[i]["d_J"] = np.dot(self.layers[i]["delta"], X.T)
            else:
                d_o_s = self.layers[i]["out"] * (1 - self.layers[i]["out"])
                self.layers[i]["delta"] = np.dot(self.layers[i+1]["weights"], self.layers[i+1]["delta"]) * d_o_s
                self.layers[i]["d_J"] = np.dot(self.layers[i-1]["out"], self.layers[i]["delta"])


    def optimizer(self):

        for layer in self.layers:
            layer["weights"] = layer["weights"] - (self.lr * layer["d_J"])


    def train(self, X, y, epochs):

        for epoch in range(0, epochs):
            self.forward(X)
            self.compute_loss(self.layers[-1]["out"], y)
            self.backward(X, y)
            self.optimizer()
            print("-"*100)
            print(f"{' '*40}EPOCH: {epoch}")
            print("-"*100)
            print(f'Loss {self.loss}')
            print(f"Out {self.layers[-1]['out']}")


