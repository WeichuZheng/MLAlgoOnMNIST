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


class SVM(AbstractMLModel):
    def __init__(self, learning_rate=0.01, lambda_param=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, x, y):
        num_samples, num_features = x.shape

        # 初始化模型参数
        self.weights = np.zeros(num_features)
        self.bias = 0

        # 梯度下降训练
        for _ in range(self.num_iterations):
            # 计算模型输出和间隔
            output = np.dot(x, self.weights) + self.bias
            margins = y * output

            # 根据间隔更新参数
            for i, margin in enumerate(margins):
                if margin >= 1:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights - np.dot(x[i], y[i]))
                    self.bias -= self.learning_rate * y[i]

    def predict(self, x):
        output = np.dot(x, self.weights) + self.bias
        return np.sign(output)

    def score(self, X, y):
        predict = self.predict(X)
        return accuracy_score(predict, y)
