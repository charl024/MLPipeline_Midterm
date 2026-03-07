from import_datasets import *
from pca_lda import *

def test():
    train_data, train_labels, test_data, test_labels = get_mnist()

    print(train_data.shape)

    r_train, r_test = get_samples_with_pca(train_data=train_data, test_data=test_data, n_components=50)

test()