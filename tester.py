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

NUM_ITER = 1024

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
    C_param_range = [0.01, 0.1, 1]
    gamma_param_range = [0.0001, 0.001, 0.01, 0.1, 1, 10]

    print(f"Number of iterations: {NUM_ITER}\n")


    for C in C_param_range:
        for gamma in gamma_param_range:
            svc = SVC(kernel='rbf', max_iter=NUM_ITER, C=C, gamma=gamma)
            pl = Pipeline(SVC=svc)
            total = pl.fit()

            train_error, test_error = calc_errors(pl)

            train_accuracy = (1.0 - train_error)
            test_accuracy  = (1.0 - test_error)

            print(f"C = {C}, gamma = {gamma}")
            print(f"training accuracy = {train_accuracy}, test accuracy = {test_accuracy}, fit time = {total}\n")

def poly_kernel_test():
    top_n = 20
    C_param_range = [0.01, 0.1, 1]
    gamma_param_range = [0.0001, 0.001, 0.01, 0.1, 1, 10]
    degree_param_range = [0, 1, 2, 3, 4, 5, 6]

    print(f"Number of iterations: {NUM_ITER}\n")

    results = []

    for C in C_param_range:
        for gamma in gamma_param_range:
            for degree in degree_param_range:
                svc = SVC(kernel='poly', max_iter=NUM_ITER, C=C, gamma=gamma, degree=degree)
                pl = Pipeline(SVC=svc)
                total = pl.fit()

                train_error, test_error = calc_errors(pl)

                train_accuracy = (1.0 - train_error)
                test_accuracy  = (1.0 - test_error)

                print(f"C = {C}, gamma = {gamma}, degree = {degree}")
                print(f"training accuracy = {train_accuracy}, test accuracy = {test_accuracy}, fit time = {total}\n")
                
                results.append((test_accuracy, train_accuracy, C, gamma, degree, total))

    print("\n\n\n")
    for test_accuracy, train_accuracy, C, gamma, degree, total in sorted(results, reverse=True)[:top_n]:
        print(f"C = {C}, gamma = {gamma}, degree = {degree}")
        print(f"train accuracy = {train_accuracy}, test accuracy = {test_accuracy}, fit time = {total}\n")

if __name__ == "__main__":
    # rbf_kernel_test()
    poly_kernel_test()