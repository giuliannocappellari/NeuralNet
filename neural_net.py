import numpy as np
from layers import *
from activation_functions import *
from loss_functions import *
from utils import *
from typing import List, Callable
from optimizers import *


class NN:

    def __init__(self, layers:List[NeuronMetaClass], lr:float=0.01, optimizer:Optimizer=Adam, loss:LossFunc=MSE) -> None:
        
        self.layers = layers
        self.lr = lr
        self.loss_value = 0
        self.loss = loss
        self.optimizer = optimizer
    

    def forward(self, x:np.array) -> None:
        
        for layer in self.layers:
            s = np.dot(layer.weights, x) + layer.bias
            x = layer.activation_func.func(s)
            layer.out = x


    def compute_loss(self, y_hat:np.array, y:np.array) -> None:

        self.loss_value = self.loss.func(y_hat, y)

    
    def backward(self, X, y) -> None:

        for i in range(len(self.layers)-1, -1, -1):
            if i == (len(self.layers) - 1):
                d_J_o = self.loss.grad(self.layers[i].out, y)
                d_o_s = self.layers[i].activation_func.grad(self.layers[i].out)
                self.layers[i].delta = d_J_o * d_o_s
                self.layers[i].d_J = np.dot(self.layers[i].delta, self.layers[i-1].out.T)
            elif i == 0:
                d_o_s = self.layers[i].activation_func.grad(self.layers[i].out)
                self.layers[i].delta = np.dot(self.layers[i+1].weights.T, self.layers[i+1].delta) * d_o_s
                self.layers[i].d_J = np.dot(self.layers[i].delta, X.T)
            else:
                d_o_s = self.layers[i].activation_func.grad(self.layers[i].out)
                self.layers[i].delta = np.dot(self.layers[i+1].weights.T, self.layers[i+1].delta) * d_o_s
                self.layers[i].d_J = np.dot(self.layers[i].delta, self.layers[i-1].out.T)


    def optimize(self, epoch:int):

        for layer in self.layers:
            self.optimizer.update(t=epoch, layer=layer)


    def train(self, X, y, epochs):

        for epoch in range(1, epochs+1):
            self.forward(X)
            self.compute_loss(self.layers[-1].out, y)
            self.backward(X, y)
            self.optimize(epoch)
            print("-"*100)
            print(f"{' '*40}EPOCH: {epoch}")
            print("-"*100)
            print(f'Loss {self.loss_value}')
            print(f"Out {self.layers[-1].out}")


x = np.array([[0.5, 0.1],[0.2, 0.6]]).T
y = [0.7, 0.8]
adam = Adam()
mse = MSE()
gd = GradientDescendent()
nn = NN(layers=[Linear(x.shape[0], 2, Sigmoid()),
                Linear(2, 3, Sigmoid()),
                Linear(3, 4, Sigmoid()),
                Linear(4, 3, Sigmoid()),
                Linear(3, 2, Sigmoid()),
                Linear(2, 1, Sigmoid()),
                ],
                optimizer=adam,
                loss=mse)
nn.train(x, y, 500)