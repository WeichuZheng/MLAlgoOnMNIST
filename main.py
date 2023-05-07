from utils import parser_mnist_data
import os
import argparse
from models import KNN
from models import SVM
from dataset import MNISTDataset
import json


def config():
    with open('./params.json') as f:
        data = json.load(f)
    return data


def run(params):
    if not os.path.exists(params["default_train_path"]):
        parser_mnist_data(params["root_data_path"])
    parser = argparse.ArgumentParser(description='test')

    parser.add_argument('--model', help='Machine Learning Model')
    parser.add_argument('--train_path', default=params["default_train_path"], help='Training data path')
    parser.add_argument('--test_path', default=params["default_test_path"], help='Testing data path')
    parser.add_argument('--dataset', default='default', help='Use default or shallow')
    parser.add_argument('--train_limit', type=int, default=params["default_train_limit"],
                        help='Train data size in each category')
    parser.add_argument('--test_limit', type=int, default=params["default_test_limit"],
                        help='Test data size in each category')
    parser.add_argument('--k', type=int, default=3, help='Hyper Parameter in knn')

    args = parser.parse_args()

    dataset = MNISTDataset(args)

    if args.model == 'knn':
        model = KNN(args=args)
    elif args.model == 'svm':
        model = SVM()
    else:
        raise NotImplementedError

    model.fit(dataset.train_data, dataset.train_labels)
    score = model.score(dataset.test_data, dataset.test_labels)
    print(score)


params = config()
run(params)
