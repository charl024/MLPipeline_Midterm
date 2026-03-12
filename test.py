from import_datasets import *
from pca_lda import *
from Pipeline import Pipeline
from dataset_reduction import *


import numpy as np
import time
import pyprind

from sklearn.datasets import make_classification
from sklearn.model_selection import validation_curve
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 

def test():
    train_data, train_labels, test_data, test_labels = get_mnist()

    # print(train_data.shape)
    # print(test_data.shape)
    # print(train_labels.shape)
    # print(test_labels.shape)

    pca_rtrain, pca_rtest = get_samples_with_pca(train_data=train_data, test_data=test_data, n_components=50)
    lda_rtrain, lda_rtest = get_samples_with_lda(train_data=train_data, train_labels=train_labels, test_data=test_data)
 
def calc_errors(pipeline):
    train_preds = pipeline.SVC.predict(pipeline.red_X_train)
    test_preds  = pipeline.predict()
    
    train_error = np.mean(train_preds != pipeline.y_train)
    test_error  = np.mean(test_preds  != pipeline.y_test)
    
    return train_error, test_error

def linear_kernel_test():
    print("Beginning")

    C_param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    train_accs = []
    test_accs = []
    train_times = []

    print("Starting now!")

    bar = pyprind.ProgBar(len(C_param_range), title='Linear Kernel')

    # print(param_range)
    for C in C_param_range:
        start = time.perf_counter() 
        svc = SVC(kernel='linear', max_iter=100, C=C)
        pl = Pipeline(SVC=svc)
        pl.fit()
        end = time.perf_counter()

        train_error, test_error = calc_errors(pl)
        total = end - start

        train_accuracy = (1.0 - train_error)
        test_accuracy  = (1.0 - test_error)

        train_accs.append(train_accuracy)
        test_accs.append(test_accuracy)

        train_times.append(total)
        bar.update()

    print(f"Train Accuracy across Cs: {train_accs}")
    print(f"Test Accuracy across Cs: {test_accs}")
    print(f"Training time across Cs: {train_times}")

linear_kernel_test()