import os
import numpy as np
from PIL import Image

class MNISTDataset():
    def __init__(self, args) -> None:
        self.train_path = args.train_path
        self.test_path = args.test_path
        self.train_limit = args.train_limit
        self.test_limit = args.test_limit
        if args.dataset == 'default':
            self.train_data, self.train_labels = self.read_data(self.train_path)
            self.test_data, self.test_labels = self.read_data(self.test_path)
        else:
            self.train_data, self.train_labels = self.read_data(self.train_path, limit=self.train_limit)
            self.test_data, self.test_labels = self.read_data(self.test_path, limit=self.test_limit)

    def read_data(self, path, limit=None):
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

