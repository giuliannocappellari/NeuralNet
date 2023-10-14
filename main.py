from neural_net import *
import requests, pickle
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

if __name__ == "__main__":


    def load_MNIST():

        with open('./data/mnist1d_data.pkl', 'rb') as handle:
            data = pickle.load(handle)

        return data['x'], data['y'], data['x_test'], data['y_test']
        

    def load_iris():

        le = LabelEncoder()
        data = pd.read_csv("data/iris.csv")
        X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
        y = le.fit_transform(data['species'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                            random_state=42, shuffle=True)
        return X_train, X_test, y_train, y_test
        

    X_train, X_test, y_train, y_test = load_iris()

    adam = Adam()
    bce = MSE()
    nn = NN(layers=[Linear(X_train.shape[1], 10, Relu(), init_mode="xavier_uniform"),
                        Linear(10, 25, Relu(), init_mode="xavier_uniform"),
                        Linear(25, 50, Relu(), init_mode="xavier_uniform"),
                        Linear(50, 100, Relu(), init_mode="xavier_uniform"),
                        Linear(100, 50, Relu(), init_mode="xavier_uniform"),
                        Linear(50, 25, Relu(), init_mode="xavier_uniform"),
                        Linear(25, 1, Relu(), init_mode="xavier_uniform"),
                        ],
                        optimizer=adam,
                        loss=MSE,
                        lr=0.001)
    
    nn.train(X_train, y_train, X_test, y_test, 1000, 50)
