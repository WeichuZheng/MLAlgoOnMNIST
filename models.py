import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score

class AbstractMLModel(object):
    def __init__(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass

    def visualization(self):
        pass


class KNN(AbstractMLModel):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def fit(self, X, y):
        self.train_data = X
        self.train_label = y

    def predict(self, X):
        self.test_data = X
        num_train = self.train_data.shape[0]
        num_test = self.test_data.shape[0]
        dists = np.zeros((num_test, num_train))

        for i in range(num_test):
            for j in range(num_train):
                print((i, j))
                diff = self.test_data[i] - self.train_data[j]
                dists[i][j] = np.sqrt(np.sum(diff ** 2))
        predicted_labels = np.zeros(num_test)
        for i in range(num_test):
            closest_k = self.train_label[np.argsort(dists[i])[:self.args.k]]
            predicted_labels[i] = np.argmax(np.bincount(closest_k))
        return predicted_labels
    
    def score(self, X, y):
        predict = self.predict(X)
        return accuracy_score(predict, y)
