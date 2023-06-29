"""
    Utility script to read parameters and data from files.
"""

import yaml
from util import *

def read_params(filepath):
    """
        Read yaml file and return a Python dictionary with all parameters.
    """
    with open(filepath) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    return params

def get_data_loaders(dataset, nodes):

    if dataset == 'CIFAR10':
        print("Dataset = ", dataset)
        print("N' of nodes = ", nodes)
        return loadCIFAR10(nodes)
    elif dataset == 'Sentiment':
        print("Dataset = ", dataset)
        print("N' of nodes = ", nodes)
        return loadSentiment(nodes)
    elif dataset == 'EMNIST':
        print("Dataset = ", dataset)
        print("N' of nodes = ", nodes)
        return loadEMNIST(nodes)
