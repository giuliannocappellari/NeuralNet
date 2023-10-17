import numpy as np
from typing import List, Callable
import pickle
import matplotlib.pyplot as plt
from layers import *
from optimizer import *
from loss_functions import *
from utils import *
from copy import deepcopy, copy


class NN:

    def __init__(self, first_layers:List[LayerMetaClass]=None,  hidden_states:Tuple[int,int,ActivationFunc]=None, 
                 final_layers:List[LayerMetaClass]=None, lr:float=0.1, optimizer:Optimizer=Adam, loss:LossFunc=MSE, 
                 batchsize:int=None, seed:int=1234, save_best_model=True) -> None:
        
        self.layers:List[LayerMetaClass] = []
        self._compile(first_layers, hidden_states, final_layers)
        self.out_train =[]
        self.out_test =[]
        self.lr = lr
        self.loss_value = []
        self.loss_value_test = []
        self.loss = loss
        self.optimizer = optimizer
        self.batch_size = batchsize
        self.train_losses = [] # para os plots
        self.test_losses = [] # para os plots
        self.train_accuracies = []
        self.test_accuracies = []
        self._best_accuracy = 0
        self.save_best_model = save_best_model


    def _compile(self, first_layers:List[LayerMetaClass]=None,  hidden_states:Tuple[int,int,ActivationFunc]=None, 
                 final_layers:List[LayerMetaClass]=None):
        
        self.layers += first_layers
        if hidden_states:
            for i in range(hidden_states[0]):
                self.layers.append(Linear(self.layers[-1].neurons, hidden_states[1], hidden_states[2]))
        self.layers += final_layers

    
    def get_model_params(self):

        print("-"*100)
        print(f"Your model have")
        print("-"*100)
        print(f"{len(self.layers)} Layers")
        print(f"{sum([layer.neurons for layer in self.layers])} Neurons")
        print(f"{sum([(layer.neurons * layer.input_size) + layer.neurons for layer in self.layers])} Params")


    def _zero_grad(self, batch_size) -> None:

        for layer in self.layers:
            layer.zero_grad(batch_size)


    def compute_loss(self, y_hat:np.array, y:np.array, train:bool=True) -> None:
        
        if train:
            self.loss_value.append(self.loss.func(y_hat, y))
        else:
            self.loss_value_test.append(self.loss.func(y_hat, y))

    
    def _optimize(self, epoch:int):

        for layer in self.layers:
            self.optimizer.update(t=epoch, layer=layer)


    def classify(self, X_to_classify):

        y_hat = []
        for batch in range(0, X_to_classify.shape[0], self.batch_size):
            X = deepcopy(X_to_classify[batch:self.batch_size+batch])
            out = self.forward(X)
            out = np.round(out)
            y_hat += out.tolist()
        return y_hat

    
    def forward(self, X:np.array, train:bool=True) -> np.array:

        for layer in self.layers:
            X = layer.forward(X)
            if train:
                self.activations.append(X)  # para o histograma
            else:
                self.activations_test.append(X)
        return X
        

    def backward(self, y:np.array, y_hat:np.array) -> None:

        dJ_ds = self.loss.grad(y, y_hat)
        for i in range(len(self.layers)-1, -1, -1):
            dJ_ds = self.layers[i].backward(dJ_ds)
            self.gradients.append(self.layers[i].d_W)

    
    def optimize(self, epoch:int):

        for layer in self.layers:
            self.optimizer.update(t=epoch, layer=layer)

    
    def train(self, X_train, y_train, X_test, y_test, epochs, batch_size):

        self._zero_grad(batch_size)
        for epoch in range(1, epochs+1):
            self.activations = []  # para plotar os histogramas
            self.activations_test = []  # para plotar os histogramas
            self.gradients = []

            for batch in range(0, X_train.shape[0], batch_size):
                y_hat = self.forward(X_train[batch:(batch+batch_size)])
                self.out_train += y_hat.tolist()
                self.compute_loss(y_hat, y_train[batch:(batch+batch_size)], train=True)
                self.backward(y_train[batch:(batch+batch_size)], y_hat)
                self.train_accuracies.append(self.calculate_accuracy(y_train[batch:(batch+batch_size)], y_hat))
            
            for batch in range(0, X_test.shape[0], batch_size):
                y_hat = self.forward(X_test[batch:(batch+batch_size)], False)
                self.out_test += y_hat.tolist()
                self.compute_loss(y_hat, y_test[batch:(batch+batch_size)], False)
                self.test_accuracies.append(self.calculate_accuracy(y_test[batch:(batch+batch_size)], y_hat))

            self.train_losses.append(sum(self.loss_value))
            self.test_losses.append(sum(self.loss_value))
            self.plot_histograms()  # histo

            self.optimize(epoch)

            train_accuracy = sum(self.train_accuracies)/len(self.train_accuracies)
            test_accuracy = sum(self.test_accuracies)/len(self.test_accuracies)

            print("-"*100)
            print(f"{' '*40}EPOCH: {epoch}")
            print("-"*100)
            print(f'Mean Loss Train {self.loss_value[-1]}')
            print(f'Mean Loss Test {self.loss_value_test[-1]}')
            print(f'Mean Train Accuracy {train_accuracy}')
            print(f'Mean Test Accuracy {test_accuracy}')

            if self.save_best_model:
                if test_accuracy > self._best_accuracy:
                    self._best_accuracy = test_accuracy
                    self.save_model("neural_net.pkl")
        
        plot_cost_vs_epoch(self)
        plot_accuracies(self)


    def calculate_accuracy(self, y, y_hat):
        predictions = np.round(y_hat)
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

            plt.show()



    def save_model(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load_model(filename):
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        model.get_model_params()
        return model