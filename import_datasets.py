import idx2numpy
import numpy as np

# standardizes train and test data
def standardize(train_data, test_data):
    train_mean = np.mean(train_data, axis=0)
    train_std = np.std(train_data, axis=0)

    train_std[train_std == 0] = 1.0

    train_standardized = (train_data - train_mean) / train_std
    test_standardized = (test_data - train_mean) / train_std
    return train_standardized, test_standardized

# returns training data/labels, testing data/labels, flattens data and standardizes
def get_fashion_mnist():
    train_images = idx2numpy.convert_from_file("./datasets/FashionMNIST/raw/train-images-idx3-ubyte")
    train_labels = idx2numpy.convert_from_file("./datasets/FashionMNIST/raw/train-labels-idx1-ubyte")
    test_images = idx2numpy.convert_from_file("./datasets/FashionMNIST/raw/t10k-images-idx3-ubyte")
    test_labels = idx2numpy.convert_from_file("./datasets/FashionMNIST/raw/t10k-labels-idx1-ubyte")

    train_flattened = train_images.reshape(train_images.shape[0], -1)
    test_flattened = test_images.reshape(test_images.shape[0], -1)

    train_standardized, test_standardized = standardize(train_flattened, test_flattened)

    return train_standardized, train_labels, test_standardized, test_labels

# returns training data/labels, testing data/labels, flattens data and standardizes
def get_mnist():
    train_images = idx2numpy.convert_from_file("./datasets/MNIST/raw/train-images-idx3-ubyte")
    train_labels = idx2numpy.convert_from_file("./datasets/MNIST/raw/train-labels-idx1-ubyte")
    test_images = idx2numpy.convert_from_file("./datasets/MNIST/raw/t10k-images-idx3-ubyte")
    test_labels = idx2numpy.convert_from_file("./datasets/MNIST/raw/t10k-labels-idx1-ubyte")

    train_flattened = train_images.reshape(train_images.shape[0], -1)
    test_flattened = test_images.reshape(test_images.shape[0], -1)

    train_standardized, test_standardized = standardize(train_flattened, test_flattened)

    return train_standardized, train_labels, test_standardized, test_labels

# print(get_fashion_mnist())
# print(get_mnist())