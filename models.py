import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image


class AbstractMLModel(object):
    def __init__(self):
        pass

    def load_data(self, path, limit):
        pass

    def fit(self):
        pass

    def visualization(self):
        pass


class KNN(AbstractMLModel):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def load_data(self, path, limit=None):
        data = []
        labels = []
        for i in range(10):
            label = i
            label_path = os.path.join(path, str(i))
            if limit:
                count = limit
                for filename in os.listdir(label_path):
                    image_path = os.path.join(label_path, filename)
                    image = Image.open(image_path)
                    image = np.array(image).flatten()
                    data.append(image)
                    labels.append(label)
                    count = count - 1
                    if count == 0:
                        break
            else:
                for filename in os.listdir(label_path):
                    image_path = os.path.join(label_path, filename)
                    image = Image.open(image_path)
                    image = np.array(image).flatten()
                    data.append(image)
                    labels.append(label)
        return np.array(data), np.array(labels)

    def fit(self):
        if self.args.dataset == 'default':
            train_data, train_labels = self.load_data(self.args.train_path)
            test_data, test_labels = self.load_data(self.args.test_path)
        else:
            train_data, train_labels = self.load_data(self.args.train_path, limit=self.args.train_limit)
            test_data, test_labels = self.load_data(self.args.test_path, limit=self.args.test_limit)
        num_train = train_data.shape[0]
        num_test = test_data.shape[0]
        dists = np.zeros((num_test, num_train))

        for i in range(num_test):
            for j in range(num_train):
                print((i, j))
                diff = test_data[i] - train_data[j]
                dists[i][j] = np.sqrt(np.sum(diff ** 2))
        predicted_labels = np.zeros(num_test)
        for i in range(num_test):
            closest_k = train_labels[np.argsort(dists[i])[:self.args.k]]
            predicted_labels[i] = np.argmax(np.bincount(closest_k))
        accuracy = np.mean(predicted_labels == test_labels)
        print(f'Accuracy: {accuracy}')
