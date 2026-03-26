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

NUM_ITER = 1

NUM_COMPONENTS = [None]

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

def update_best(arr, x, C, time, gamma=0, degree=0):
    """
    Update and return array, based on if x value and arr[0]
    """
    if (arr[1] < x):
        arr[0] = time
        arr[1] = x
        arr[2] = C
        arr[3] = gamma
        arr[4] = degree
    return arr

def single_test(svc, num_comp, dataset_str):
    """
    Creates a single model, times it, tracks error
    """
    pl = Pipeline(SVC=svc, num_components=num_comp, preprocessor="lda", data_set=dataset_str)
    time = pl.fit()

    train_error, test_error = calc_errors(pl)
    train_accuracy = (1.0 - train_error)
    test_accuracy  = (1.0 - test_error)

    # Print accuracies
    print(f"training accuracy = {train_accuracy}, test accuracy = {test_accuracy}, fit time = {time}\n")

    return test_accuracy, time

def linear_kernel_test(dataset_str):
    C_param_range = [0.0001, 0.001, 0.01, 0.1, 1]

    print(f"Number of iterations: {NUM_ITER}\n")

    for num_comp in NUM_COMPONENTS:
        print(f"Number of components for PCA reduction = {num_comp}\n")

        test_best = [0, 0, 0, 0, 0]

        for C in C_param_range:
            # Print hyper params
            print(f"C = {C}")

            # Make Model
            svc = SVC(kernel='linear', max_iter=NUM_ITER, C=C)
            test_acc, test_time = single_test(svc=svc, num_comp=num_comp, dataset_str=dataset_str)

            # Track the best test accuracy
            test_best = update_best(test_best, test_acc, C, time=test_time)
        
        #OUTPUT BEST on test data set
        print(f"Test Best: C = {test_best[2]}, test accuracy: {test_best[1]}, time: {test_best[0]}\n")

def rbf_kernel_test(dataset_str):
    C_param_range = [0.1, 1, 10, 100, 1000]
    gamma_param_range = [0.0001, 0.001, 0.01, 0.1, 1]

    print(f"Number of iterations: {NUM_ITER}\n")

    for num_comp in NUM_COMPONENTS:
        print(f"Number of components for PCA reduction = {num_comp}\n")

        test_best = [0, 0, 0, 0, 0]

        for C in C_param_range:
            for gamma in gamma_param_range:
                # Print hyper params
                print(f"C = {C}, gamma = {gamma}")

                # Make Model
                svc = SVC(kernel='rbf', max_iter=NUM_ITER, C=C, gamma=gamma)
                test_acc, test_time = single_test(svc=svc, num_comp=num_comp, dataset_str=dataset_str)
                # Track the best test accuracy
                test_best = update_best(test_best, test_acc, C, gamma, time=test_time)
        
        #OUTPUT BEST on test data set
        print(f"Test Best: C = {test_best[2]}, gamma = {test_best[3]}, test accuracy: {test_best[1]}, time: {test_best[0]}\n")

def poly_kernel_test(dataset_str):
    # C = 0.1, gamma = 0.01, degree = 3 did best
    C_param_range = [0.001, 0.01, 0.1, 1, 10]
    gamma_param_range = [0.0001, 0.001, 0.01, 0.1, 1]
    degree_param_range = [1, 2, 3, 4, 5]

    print(f"Number of iterations: {NUM_ITER}\n")

    for num_comp in NUM_COMPONENTS:
        print(f"Number of components for PCA reduction = {num_comp}\n")

        test_best = [0, 0, 0, 0, 0]

        for C in C_param_range:
            for gamma in gamma_param_range:
                for degree in degree_param_range:
                    # Print hyper params
                    print(f"C = {C}, gamma = {gamma}, degree = {degree}")

                    # Make Model
                    svc = SVC(kernel='poly', 
                              max_iter=NUM_ITER, 
                              C=C, gamma=gamma, 
                              degree=degree)
                    
                    test_acc, test_time = single_test(svc=svc, num_comp=num_comp, dataset_str=dataset_str)

                    # Track the best test accuracy
                    test_best = update_best(test_best, 
                                            test_acc, 
                                            C=C, 
                                            gamma=gamma, 
                                            degree=degree, time=test_time)
        
        #OUTPUT BEST on test data set
        print(f"Test Best: C = {test_best[2]}, gamma = {test_best[3]}, degree = {test_best[4]}, test accuracy: {test_best[1]}, time: {test_best[0]}\n")

if __name__ == "__main__":
    print("LINEAR KERNEL WITH LDA-MNIST")
    linear_kernel_test("mnist")
    print("RBF KERNEL WITH LDA-MNIST")
    rbf_kernel_test("mnist")
    print("POLY KERNEL WITH LDA-MNIST")
    poly_kernel_test("mnist")

    print("LINEAR KERNEL WITH LDA-FASHIONMNIST")
    linear_kernel_test("fashionmnist")
    print("RBF KERNEL WITH LDA-FASHIONMNIST")
    rbf_kernel_test("mnist")
    print("POLY KERNEL WITH LDA-FASHIONMNIST")
    poly_kernel_test("fashionmnist")
