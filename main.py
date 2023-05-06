from utils import parser_mnist_data
import os
import argparse
from models import KNN


def run():
    if not os.path.exists('./data/train'):
        parser_mnist_data('./data/')
    parser = argparse.ArgumentParser(description='test')

    parser.add_argument('--model', help='Machine Learning Model')
    parser.add_argument('--train_path', default='./data/train/', help='Training data path')
    parser.add_argument('--test_path', default='./data/test/', help='Testing data path')
    parser.add_argument('--dataset', default='default', help='Use default or shallow')
    parser.add_argument('--train_limit', type=int, default=300, help='Train data size in each category')
    parser.add_argument('--test_limit', type=int, default=50, help='Test data size in each category')
    parser.add_argument('--k', type=int, default=3, help='Hyper Parameter in knn')

    args = parser.parse_args()

    if args.model == 'knn':
        model = KNN(args=args)
    else:
        raise NotImplementedError

    model.fit()


run()
