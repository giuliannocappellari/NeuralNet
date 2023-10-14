import numpy as np
from layers import *
from activation_functions import *
from loss_functions import *
from utils import *
from typing import List, Callable
from optimizers import *
import pickle
import matplotlib.pyplot as plt

class NN:

    def __init__(self, layers:List[NeuronMetaClass], lr:float=0.1, optimizer:Optimizer=Adam, loss:LossFunc=MSE, batchsize:int=None) -> None:
        
        self.layers = layers
        self.lr = lr
        self.loss_value = []
        self.loss_value_test = []
        self.loss = loss
        self.optimizer = optimizer
        self.batch_size = batchsize

        self.activations = []  # para plotar os histogramas
        self.activations_test = []  # para plotar os histogramas

        self.train_losses = [] # para os plots
        self.test_losses = [] # para os plots

        self.train_accuracies = []
        self.test_accuracies = []
        


    def zero_grad(self, input_shape):
        self.loss_value = []
        self.loss_value_test = []
        for layer in self.layers:
            layer.zero_grad(input_shape)


    def forward(self, x:np.array) -> None:
        for layer in self.layers:
            s = np.dot(layer.weights, x) + layer.bias
            x = layer.activation_func.func(s)
            layer.out = x
            self.activations.append(x)  # para o histograma
    

    def forward_test(self, x:np.array) -> None:
        self.activations = []  # para plotar os histogramas
        for layer in self.layers:
            s = np.dot(layer.weights, x) + layer.bias
            x = layer.activation_func.func(s)
            layer.out = x
            self.activations_test.append(x)  # para o histograma
        


    def compute_loss(self, y_hat:np.array, y:np.array) -> None:

        self.loss_value.append(self.loss.func(y_hat, y))

    
    def compute_loss_test(self, y_hat:np.array, y:np.array) -> None:

        self.loss_value_test.append(self.loss.func(y_hat, y))

    
    def backward(self, X, y) -> None:
        self.gradients = []  # histogramas
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
            self.gradients.append(self.layers[i].delta)  # histogramas

    def optimize(self, epoch:int):

        for layer in self.layers:
            self.optimizer.update(t=epoch, layer=layer)


    def train(self, X_train, y_train, X_test, y_test, epochs, batch_size):
        for epoch in range(1, epochs+1):
            self.zero_grad(batch_size)
            for batch in range(0, X_train.shape[1], batch_size):
                self.forward(X_train[batch:(batch+batch_size)].T)
                self.compute_loss(self.layers[-1].out, y_train[batch:(batch+batch_size)])
                self.backward(X_train[batch:(batch+batch_size)].T, y_train[batch:(batch+batch_size)])
                self.train_accuracies.append(self.calculate_accuracy(y_train[batch:(batch+batch_size)]))
                # self.plot_histograms()  # histo
            # Avalie no conjunto de validação
            for batch in range(0, X_test.shape[1], batch_size):
                self.forward_test(X_test[batch:(batch+batch_size)].T)
                self.compute_loss_test(self.layers[-1].out, y_test[batch:(batch+batch_size)])
                self.test_accuracies.append(self.calculate_accuracy(y_test[batch:(batch+batch_size)]))

            # Após treinar no conjunto de treino
            self.train_losses.append(sum(self.loss_value))
            self.test_losses.append(sum(self.loss_value))

            self.optimize(epoch)

            print("-"*100)
            print(f"{' '*40}EPOCH: {epoch}")
            print("-"*100)
            print(f'Mean Loss Train {sum(self.loss_value)/len(self.loss_value)}')
            print(f'Mean Loss Test {sum(self.loss_value_test)/len(self.loss_value_test)}')
            print(f'Mean Train Accuracy {sum(self.train_accuracies)/len(self.train_accuracies)}')
            print(f'Mean Test Accuracy {sum(self.test_accuracies)/len(self.test_accuracies)}')


    def calculate_accuracy(self, y):
        predictions = np.round(self.layers[-1].out)
        accuracy = np.mean(predictions == y)
        return accuracy

    def plot_histograms(self):
        for i, (activation, gradient) in enumerate(zip(self.activations, self.gradients)):
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.hist(activation.flatten(), bins=30, alpha=0.75)
            plt.title(f'Layer {i + 1} Activations')

            plt.subplot(1, 2, 2)
            plt.hist(gradient.flatten(), bins=30, alpha=0.75, color='red')
            plt.title(f'Layer {i + 1} Gradients')

            plt.tight_layout()

            plt.pause(0.5)  # Exibe o gráfico por 0.5 segundos
            plt.close() #ao invés de plt.show()

            #plt.show()



    def save_model(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load_model(filename):
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        return model


if __name__ == "__main__":
    x = np.array([[0.5, 0.1],[0.2, 0.6],[0.5, 0.1],[0.2, 0.6], [0.5, 0.1],[0.2, 0.6],
                [0.5, 0.1],[0.2, 0.6], [0.5, 0.1],[0.2, 0.6], [0.5, 0.1],[0.2, 0.6]])

    y = [0.7, 0.8, 0.7, 0.8, 0.7, 0.8, 0.7, 0.8, 0.7, 0.8, 0.7, 0.8]

    x_test = np.array([[0.5, 0.1],[0.2, 0.6],[0.5, 0.1],[0.2, 0.6], [0.5, 0.1],[0.2, 0.6]])

    y_test = [0.7, 0.8, 0.7, 0.8, 0.7, 0.8]

    adam = Adam()
    gd = GradientDescendent()
    mse = MSE()
    bce = BinaryCrossEntropy()
    funcs = [Relu]

    for func in funcs:
        # loaded_nn = NN.load_model('model.pkl')
        nn = NN(layers=[Linear(x.shape[1], 2, func(), init_mode="xavier_uniform"),
                        Linear(2, 3, func(), init_mode="xavier_uniform"),
                        Linear(3, 4, func(), init_mode="xavier_uniform"),
                        Linear(4, 3, func(), init_mode="xavier_uniform"),
                        Linear(3, 2, func(), init_mode="xavier_uniform"),
                        Linear(2, 1, func(), init_mode="xavier_uniform"),
                        ],
                        optimizer=adam,
                        loss=mse)
        nn.train(x, y, x_test, y_test, 120, 12)
        nn.save_model('model.pkl')

        # plot_accuracies(nn)

        # Após treinar a rede neural
        # plot_cost_vs_epoch(nn)