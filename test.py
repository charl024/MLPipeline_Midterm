from import_datasets import *
from pca_lda import *

def test():
    train_data, train_labels, test_data, test_labels = get_mnist()

    print(train_data.shape)
    print(test_data.shape)
    print(train_labels.shape)
    print(test_labels.shape)

    pca_rtrain, pca_rtest = get_samples_with_pca(train_data=train_data, test_data=test_data, n_components=50)
    lda_rtrain, lda_rtest = get_samples_with_lda(train_data=train_data, train_labels=train_labels, test_data=test_data)

test()