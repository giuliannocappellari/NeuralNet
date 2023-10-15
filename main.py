from neural_net import *
import requests, pickle
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

if __name__ == "__main__":


    def load_MNIST():

        with open('./data/mnist1d_data.pkl', 'rb') as handle:
            data = pickle.load(handle)

        data['x'] = data['x']/255
        data['x_test'] = data['x_test']/255

        return data['x'], data['x_test'], data['y'], data['y_test']
        

    def load_iris():

        le = LabelEncoder()
        encoder = OneHotEncoder(sparse=False)
        data = pd.read_csv("data/iris.csv")
        X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
        y = le.fit_transform(data['species'])
        # y = encoder.fit_transform(data['species'].values.reshape(data['species'].values.shape[0], 1))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                            random_state=42, shuffle=True)
        return X_train, X_test, y_train, y_test
    
        

    X_train, X_test, y_train, y_test = load_iris()
    # X_train, X_test, y_train, y_test = load_MNIST()

    adam = Adam()
    gd = GradientDescendent()
    cel = CrossEntropyLoss
    mse = MSE
    bce = BinaryCrossEntropy
    nn = NN(first_layers=[Linear(X_train.shape[1], 32, Relu(), init_mode="xavier_uniform"),
                        Linear(32, 16, Relu(), init_mode="xavier_uniform"),
                        ],
            final_layers=[Linear(16, 1, Relu(), init_mode="xavier_uniform"),
            ],
            optimizer=adam,
            loss=mse(),
            lr=0.001)
    
    nn.train(X_train, y_train, X_test, y_test, 54, 10)

    # print(teste[0] != teste[1])
