from import_datasets import get_fashion_mnist, get_mnist
from pca_lda import get_samples_with_lda, get_samples_with_pca
import numpy as np

def reduce_dataset_and_save(train_data, train_labels, test_data, test_labels, filename):
    pca_rtrain50, pca_rtest50 = get_samples_with_pca(train_data=train_data, test_data=test_data, n_components=50)
    pca_rtrain100, pca_rtest100 = get_samples_with_pca(train_data=train_data, test_data=test_data, n_components=100)
    pca_rtrain200, pca_rtest200 = get_samples_with_pca(train_data=train_data, test_data=test_data, n_components=200)
    lda_rtrain, lda_rtest = get_samples_with_lda(train_data=train_data, train_labels=train_labels, test_data=test_data)
    np.savez(
        "./datasets_reduced/" + filename + ".npz",
        pca_train50=pca_rtrain50,
        pca_test50=pca_rtest50,
        pca_train100=pca_rtrain100,
        pca_test100=pca_rtest100,
        pca_train200=pca_rtrain200,
        pca_test200=pca_rtest200,
        lda_train=lda_rtrain,
        lda_test=lda_rtest,
        train_labels=train_labels,
        test_labels=test_labels
    )

def load_dataset_from_file(filename, num_components=None):
    loaded_data = np.load("./datasets_reduced/" + filename + ".npz")

    pca_rtrain50 = loaded_data["pca_train50"]
    pca_rtest50 = loaded_data["pca_test50"]
    pca_rtrain100 = loaded_data["pca_train100"]
    pca_rtest100 = loaded_data["pca_test100"]
    pca_rtrain200 = loaded_data["pca_train200"]
    pca_rtest200 = loaded_data["pca_test200"]

    lda_rtrain = loaded_data["lda_train"]
    lda_rtest = loaded_data["lda_test"]
    train_labels = loaded_data["train_labels"]
    test_labels = loaded_data["test_labels"]

    if num_components == None:
        return lda_rtrain, train_labels, lda_rtest, test_labels
    elif num_components == 50:
        return pca_rtrain50, train_labels, pca_rtest50, test_labels
    elif num_components == 100:
        return pca_rtrain100, train_labels, pca_rtest100, test_labels
    elif num_components == 200:
        return pca_rtrain200, train_labels, pca_rtest200, test_labels

def save_mnist():
    train_data, train_labels, test_data, test_labels = get_mnist()
    reduce_dataset_and_save(train_data, train_labels, test_data, test_labels, "mnist_reduced")


def save_fashion_mnist():
    train_data, train_labels, test_data, test_labels = get_fashion_mnist()
    reduce_dataset_and_save(train_data, train_labels, test_data, test_labels, "fashionmnist_reduced")

# train_data, train_labels, test_data, test_labels = load_dataset_from_file("mnist_reduced", None)