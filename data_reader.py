import os
import sys
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch

def take_average(arr, every_n):
    """
    The function that takes the average of every n elements in the array
    :param: The array to be averaged
    :param: every_n: average every n elements
    """
    l, w = np.shape(arr)
    arr_avg = np.mean(arr.reshape([l, -1, every_n]), axis=2)
    print('before take average', np.shape(arr))
    print('after take average', np.shape(arr_avg))
    return arr_avg

def get_data_into_loaders(data_x, data_y,  batch_size, DataSetClass, rand_seed=1, test_ratio=0.05, average=None, square=False):
    """
    Helper function that takes structured data_x and data_y into dataloaders
    :param data_x: the structured x data
    :param data_y: the structured y data
    :param rand_seed: the random seed
    :param test_ratio: The testing ratio
    :return: train_loader, test_loader: The pytorch data loader file
    """
    # Normalize the input
    # x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=test_ratio,
    #                                                    random_state=rand_seed)
    
    # new shuffle scheme to account for the bruce dataset
    train_test_bound = 1237
    
    # Square it for the sake of volatility analysis
    if square:
        data_x = np.square(data_x)
        print('the dataset is squared')
        assert np.min(data_x) > 0, 'the dataset is squared and therefore the minimum value should be larger than 0'

    x_train, y_train = data_x[:train_test_bound, :], data_y[:train_test_bound]
    x_test, y_test = data_x[train_test_bound:, :], data_y[train_test_bound:]
    
    # Get the length of the training data
    train_len, test_len = len(x_train), len(x_test)
    
    # Shuffle them
    train_shuffle_index = np.random.permutation(train_len)
    test_shuffle_index = np.random.permutation(test_len)

    x_train = x_train[train_shuffle_index]
    y_train = y_train[train_shuffle_index]
    x_test = x_test[test_shuffle_index]
    y_test = y_test[test_shuffle_index]
    
    # This step is to get the averaged data for 5/10/15 minutes
    if average is not None:
        x_train = take_average(x_train, average)
        x_test = take_average(x_test, average)
    
    print('total number of training sample is {}, the dimension of the feature is {}'.format(len(x_train),
                                                                                             len(x_train[0])))
    print('total number of test sample is {}'.format(len(y_test)))

    # Construct the dataset using a outside class
    train_data = DataSetClass(x_train, y_train)
    test_data = DataSetClass(x_test, y_test)

    # Construct train_loader and test_loader
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    return train_loader, test_loader


def normalize_np(x):
    """
    Normalize the x into [-1, 1] range in each dimension [:, i]
    :param x: np array to be normalized
    :return: normalized np array
    """
    for i in range(len(x[0])):
        x_max = np.max(x[:, i])
        x_min = np.min(x[:, i])
        x_range = (x_max - x_min) / 2.
        x_avg = (x_max + x_min) / 2.
        x[:, i] = (x[:, i] - x_avg) / x_range
        print("In normalize_np, row ", str(i), " your max is:", np.max(x[:, i]))
        print("In normalize_np, row ", str(i), " your min is:", np.min(x[:, i]))
        assert np.max(x[:, i]) - 1 < 0.0001, 'your normalization is wrong'
        assert np.min(x[:, i]) + 1 < 0.0001, 'your normalization is wrong'
    return x


def read_data_bruce(flags, eval_data_all=False):
    # Read the data
    data_dir = '/home/sr365/Bruce/cvdata'
    print("data_dir = ", data_dir)
    data = np.load(os.path.join(data_dir, str(flags.comp_ind) + '.npy'))
    data = data.astype(np.float32)
    data_x = data[:, 1:]
    data_y = data[:, 0]
    if eval_data_all:
        return get_data_into_loaders(data_x, data_y, flags.batch_size, 
                SimulatedDataSet_xd_to_1d_class, test_ratio=0.98)
    return get_data_into_loaders(data_x, data_y, flags.batch_size, SimulatedDataSet_xd_to_1d_class,
                                 test_ratio=flags.test_ratio, average=flags.average, square=flags.square)


def read_data(flags, eval_data_all=False):
    """
    The data reader allocator function
    The input is categorized into couple of different possibilities
    :param flags: The input flag of the input data set
    :param eval_data_all: The switch to turn on if you want to put all data in evaluation data
    :return:
    """
    print("In read_data, flags.data_set =", flags.data_set)
    if 'bruce' in flags.data_set or 'Bruce' in flags.data_set:
        train_loader, test_loader = read_data_bruce(flags, eval_data_all=eval_data_all)
    else:
        sys.exit("Your flags.data_set entry is not correct, check again!")
    return train_loader, test_loader


class MetaMaterialDataSet(Dataset):
    """ The Meta Material Dataset Class """

    def __init__(self, ftr, lbl, bool_train):
        """
        Instantiate the Dataset Object
        :param ftr: the features which is always the Geometry !!
        :param lbl: the labels, which is always the Spectra !!
        :param bool_train:
        """
        self.ftr = ftr
        self.lbl = lbl
        self.bool_train = bool_train
        self.len = len(ftr)

    def __len__(self):
        return self.len

    def __getitem__(self, ind):
        return self.ftr[ind, :], self.lbl[ind, :]


class SimulatedDataSet_class_1d_to_1d(Dataset):
    """ The simulated Dataset Class for classification purposes"""

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.len = len(x)

    def __len__(self):
        return self.len

    def __getitem__(self, ind):
        return self.x[ind], self.y[ind]


class SimulatedDataSet_xd_to_1d_class(Dataset):
    """ The simulated Dataset Class for classification purposes"""

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.len = len(x)

    def __len__(self):
        return self.len

    def __getitem__(self, ind):
        return self.x[ind, :], self.y[ind]


class SimulatedDataSet_regress(Dataset):
    """ The simulated Dataset Class for regression purposes"""

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.len = len(x)

    def __len__(self):
        return self.len

    def __getitem__(self, ind):
        return self.x[ind, :], self.y[ind, :]
