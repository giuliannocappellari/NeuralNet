import numpy as np
import matplotlib.pyplot as plt


def normal_dist(x:float, sd, mean:float=0):
    prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density


def plot_cost_vs_epoch(nn):
        epochs = range(1, len(nn.train_losses) + 1)
        plt.plot(epochs, nn.train_losses, label='Treino')
        plt.plot(epochs, nn.test_losses, label='Validação')
        plt.xlabel('Épocas')
        plt.ylabel('Função de Custo')
        plt.title('Função de Custo vs Época')
        plt.legend()
        plt.show()

def plot_accuracies(nn):
    epochs = range(1, len(nn.train_accuracies) + 1)
    plt.plot(epochs, nn.train_accuracies, '-o', label='Training Accuracy')
    plt.plot(epochs, nn.test_accuracies, '-o', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()