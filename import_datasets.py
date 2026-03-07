import idx2numpy

# returns training data/labels, testing data/labels, flattens data
def get_fashion_mnist():
    train_images = idx2numpy.convert_from_file("./datasets/FashionMNIST/raw/train-images-idx3-ubyte")
    train_labels = idx2numpy.convert_from_file("./datasets/FashionMNIST/raw/train-labels-idx1-ubyte")
    test_images = idx2numpy.convert_from_file("./datasets/FashionMNIST/raw/t10k-images-idx3-ubyte")
    test_labels = idx2numpy.convert_from_file("./datasets/FashionMNIST/raw/t10k-labels-idx1-ubyte")

    train_flattened = train_images.reshape(train_images.shape[0], -1)
    test_flattened = test_images.reshape(test_images.shape[0], -1)

    return train_flattened, train_labels, test_flattened, test_labels

def get_mnist():
    train_images = idx2numpy.convert_from_file("./datasets/MNIST/raw/train-images-idx3-ubyte")
    train_labels = idx2numpy.convert_from_file("./datasets/MNIST/raw/train-labels-idx1-ubyte")
    test_images = idx2numpy.convert_from_file("./datasets/MNIST/raw/t10k-images-idx3-ubyte")
    test_labels = idx2numpy.convert_from_file("./datasets/MNIST/raw/t10k-labels-idx1-ubyte")

    train_flattened = train_images.reshape(train_images.shape[0], -1)
    test_flattened = test_images.reshape(test_images.shape[0], -1)

    return train_flattened, train_labels, test_flattened, test_labels

# print(get_fashion_mnist())
# print(get_mnist())