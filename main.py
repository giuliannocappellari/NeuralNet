from neural_net import *
import requests, pickle
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
from activation_functions import *

if __name__ == "__main__":


    def load_MNIST(onehot:bool=True):

        with open('../DL1/data/mnist1d_data.pkl', 'rb') as handle:
            data = pickle.load(handle)

        data['x'] = data['x']/255
        data['x_test'] = data['x_test']/255

        if onehot:
            encoder = OneHotEncoder(sparse=False)
            data['y'] = encoder.fit_transform(data['y'].reshape(-1, 1))
            data['y_test'] = encoder.fit_transform(data['y_test'].reshape(-1, 1))

        return data['x'], data['x_test'], data['y'], data['y_test']
        

    def load_iris(onehot:bool=True):

        data = pd.read_csv("../DL1/data/iris.csv")
        
        if onehot:
            encoder = OneHotEncoder(sparse=False)
        else:
            encoder = LabelEncoder()
        
        X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
        y = encoder.fit_transform(data['species'].values.reshape(data['species'].values.shape[0], 1))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                            random_state=42, shuffle=True)
        return X_train, X_test, y_train, y_test
    


    X_train, X_test, y_train, y_test = load_iris()
    # X_train, X_test, y_train, y_test = load_MNIST(True)
    adam = Adam()
    gd = GradientDescendent()
    cel = CrossEntropyLoss()
    mse = MSE()
    bce = BinaryCrossEntropy()

    nn = NN(first_layers=[
                        Linear(X_train.shape[1], 16, Relu(), initialize_weights=XavierNormal),
                        Linear(16, 50, Relu(), initialize_weights=XavierNormal),
                        ],
                        # hidden_states=(50,50, Sigmoid()),
            final_layers=[Linear(50, 25, Relu(), initialize_weights=XavierNormal),
                          Linear(25, 10, Relu(), initialize_weights=XavierNormal),
                          Linear(10, 3, Sigmoid(), initialize_weights=XavierNormal),
            ],
            optimizer=adam,
            loss=mse,
            lr=0.001, 
            save_best_model=True,
            model_name='iris_model.pkl')
    
    # nn = NN(first_layers=[
    #                     Linear(X_train.shape[1], 16, Relu(), initialize_weights=XavierNormal),
    #                     Linear(16, 50, Sigmoid(), initialize_weights=XavierNormal),
    #                     ],
    #                     # hidden_states=(50,50, Sigmoid()),
    #         final_layers=[Linear(50, 25, Sigmoid(), initialize_weights=XavierNormal),
    #                       Linear(25, 10, Sigmoid(), initialize_weights=XavierNormal),
    #                       Linear(10, 10, Sigmoid(), initialize_weights=XavierNormal),
    #         ],
    #         optimizer=adam,
    #         loss=mse,
    #         lr=0.001)
    
    nn.train(X_train, y_train, X_test, y_test, 1000, X_test.shape[0])

    # nn = NN.load_model("neural_net.pkl")
    # nn.batch_size = 10

    # print(nn.classify(X_test[-10:]))
    # print(nn.classify(X_test))    
    # print(nn._best_accuracy)
    # print(np.mean(y_test == nn.classify(X_test)))
    # print(nn.layers[-1].weights)


    # print(teste[0] != teste[1])
