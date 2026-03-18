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

NUM_ITER = 8192

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
    C_param_range = [0.0001, 0.001, 0.01, 0.1, 1]

    print(f"Number of iterations: {NUM_ITER}\n")


    for C in C_param_range:
        svc = SVC(kernel='linear', max_iter=NUM_ITER, C=C)
        pl = Pipeline(SVC=svc)
        total = pl.fit()

        train_error, test_error = calc_errors(pl)

        train_accuracy = (1.0 - train_error)
        test_accuracy  = (1.0 - test_error)

        print(f"C = {C}")
        print(f"training accuracy = {train_accuracy}, test accuracy = {test_accuracy}, fit time = {total}\n")

def rbf_kernel_test():
    C_param_range = [0.1, 1, 10, 100, 1000]
    gamma_param_range = [0.0001, 0.001, 0.01, 0.1, 1]

    print(f"Number of iterations: {NUM_ITER}\n")
        
    train_best = [0, 0, 0]
    test_best = [0, 0, 0]
    best = [0, 0, 0]


    for C in C_param_range:
        for gamma in gamma_param_range:
            svc = SVC(kernel='rbf', max_iter=NUM_ITER, C=C, gamma=gamma)
            pl = Pipeline(SVC=svc)
            total = pl.fit()

            train_error, test_error = calc_errors(pl)

            train_accuracy = (1.0 - train_error)
            test_accuracy  = (1.0 - test_error)

            if (train_best[0] < train_accuracy):
                train_best[0] = train_accuracy
                train_best[1] = C
                train_best[2] = gamma

            if (test_best[0] < test_accuracy):
                test_best[0] = test_accuracy
                test_best[1] = C
                test_best[2] = gamma

            if (best[0] < (test_accuracy + train_accuracy) / 2):
                best[0] = (test_accuracy + train_accuracy) / 2
                best[1] = C
                best[2] = gamma

            print(f"C = {C}, gamma = {gamma}")
            print(f"training accuracy = {train_accuracy}, test accuracy = {test_accuracy}, fit time = {total}\n")
    print(f"Train Best: C = {train_best[1]}, gamma = {train_best[2]}, training accuracy: {train_best[0]}")
    print(f"Test Best: C = {test_best[1]}, gamma = {test_best[2]}, test accuracy: {test_best[0]}")
    print(f"Best Average: C = {best[1]}, gamma = {best[2]}, (test + train) / 2 : {best[0]}")

rbf_kernel_test()
