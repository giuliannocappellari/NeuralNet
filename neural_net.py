import numpy as np
from layers import *
from activation_functions import *
from loss_functions import *
from utils import *
from typing import List, Callable
from optimizers import *


class NN:

    def __init__(self, layers:List[NeuronMetaClass], lr:float=0.01, optimizer:Optimizer=Adam, loss:LossFunc=MSE, batchsize:int=None) -> None:
        
        self.layers = layers
        self.lr = lr
        self.loss_value = 0
        self.loss = loss
        self.optimizer = optimizer
        self.batch_size = batchsize


    def zero_grad(self, input_shape):
        for layer in self.layers:
            layer.zero_grad(input_shape)
    

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
                self.layers[i].delta = self.layers[i].activation_func.get_delta(d_J_o, d_o_s) + self.layers[i].delta
                self.layers[i].d_J = np.dot(self.layers[i].delta, self.layers[i-1].out.T) + self.layers[i].d_J
                self.layers[i].d_B = np.sum(self.layers[i].delta)
            elif i == 0:
                d_o_s = self.layers[i].activation_func.grad(self.layers[i].out)
                self.layers[i].delta = self.layers[i].activation_func.get_delta(np.dot(self.layers[i+1].weights.T, self.layers[i+1].delta), d_o_s) + self.layers[i].delta
                self.layers[i].d_J = np.dot(self.layers[i].delta, X.T) + self.layers[i].d_J
                self.layers[i].d_B = np.sum(self.layers[i].delta)
            else:
                d_o_s = self.layers[i].activation_func.grad(self.layers[i].out)
                self.layers[i].delta = self.layers[i].activation_func.get_delta(np.dot(self.layers[i+1].weights.T, self.layers[i+1].delta), d_o_s) + self.layers[i].delta
                self.layers[i].d_J = np.dot(self.layers[i].delta, self.layers[i-1].out.T) + self.layers[i].d_J
                self.layers[i].d_B = np.sum(self.layers[i].delta)


    def optimize(self, epoch:int):

        for layer in self.layers:
            self.optimizer.update(t=epoch, layer=layer)


    def train(self, X, y, epochs, batch_size):
        for epoch in range(1, epochs+1):
            self.zero_grad(batch_size)
            for batch in range(0, X.shape[1], batch_size):
                self.forward(X[batch:(batch+batch_size)].T)
                self.compute_loss(self.layers[-1].out, y[batch:(batch+batch_size)])
                self.backward(X[batch:(batch+batch_size)].T, y[batch:(batch+batch_size)])
            self.optimize(epoch)
            print("-"*100)
            print(f"{' '*40}EPOCH: {epoch}")
            print("-"*100)
            print(f'Mean Loss {sum(self.loss_value[0])/self.loss_value[0].shape[0]}')
            print(f"Out {self.layers[-1].out}")


x = np.array([[0.5, 0.1],[0.2, 0.6],[0.5, 0.1],[0.2, 0.6], [0.5, 0.1],[0.2, 0.6],
              [0.5, 0.1],[0.2, 0.6], [0.5, 0.1],[0.2, 0.6], [0.5, 0.1],[0.2, 0.6]])
y = [0.7, 0.8, 0.7, 0.8, 0.7, 0.8, 0.7, 0.8, 0.7, 0.8, 0.7, 0.8, 0.7, 0.8, 0.7, 0.8,
     0.7, 0.8, 0.7, 0.8, 0.7, 0.8, 0.7, 0.8]
# x = np.array([[0.5, 0.1],[0.2, 0.6],[0.5, 0.1],[0.2, 0.6], [0.5, 0.1],[0.2, 0.6],
#               [0.5, 0.1],[0.2, 0.6], [0.5, 0.1],[0.2, 0.6], [0.5, 0.1],[0.2, 0.6]])
# y = [7,8,7,8,7,8,7,8,7,8,7,8,7,8,7,8,7,8,7,8,7,8,7,8]
adam = Adam()
gd = GradientDescendent()
mse = MSE()
bce = BinaryCrossEntropy()
funcs = [LinearActivation, Sigmoid, Tanh, Relu, LeakyRelu]
for func in funcs:
    nn = NN(layers=[Linear(x.shape[1], 2, func()),
                    Linear(2, 3, func()),
                    Linear(3, 4, func()),
                    Linear(4, 3, func()),
                    Linear(3, 2, func()),
                    Linear(2, 1, func()),
                    ],
                    optimizer=adam,
                    loss=mse)
    nn.train(x, y, 2, 12)