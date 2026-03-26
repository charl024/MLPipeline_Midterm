"""
A pipeline class for testing various models
"""
from dataset_reduction import load_dataset_from_file
from dataset_reduction import save_mnist
from dataset_reduction import save_fashion_mnist
from sklearn.svm import SVC 
import import_datasets
import pca_lda
import time

MNST_STR = "mnist"

class Pipeline:
    def __init__(self,
                 data_set=MNST_STR,
                 preprocessor="pca", 
                 SVC=SVC(max_iter=10),
                 num_components=50):
        """
        A Pipeline object
        data_set       - "mnist" for mnist, fashion otw
        preprocessor   - "pca" for PCA, LDA otw
        num_components - only relevant for PCA preprocessing (None for LDA)
        X_test         - the test data
        """
        
        #Define dataset function
        self.data_set = data_set

        #Define the preprocessing strategy
        if preprocessor == "pca":
            self.preprocessor = pca_lda.get_samples_with_pca
            self.num_components=num_components
        else:
            self.preprocessor = pca_lda.get_samples_with_lda
            self.num_components = None

        # SVC
        self.SVC = SVC

        # Load in preprocessed data
        self._transform()


    def _transform(self):
        """
        Perform preprocessing on X_train and X_test
        """

        red_X_test = None
        red_X_train = None
        try:
            if (self.data_set == MNST_STR):
                filename = "mnist_reduced"
            else:
                filename = "fashionmnist_reduced"

            red_X_train, y_train, red_X_test, y_test = load_dataset_from_file(filename=filename, num_components=self.num_components)

        except ValueError as e:
            print(f"An exception occurred: {e}")
            
        #TODO: file reading instead of function calls every time

        if (red_X_train is None or red_X_test is None):
            print("loading in datasets")
            save_mnist()
            save_fashion_mnist()
            if (self.data_set == MNST_STR):
                filename = "mnist_reduced"
            else:
                filename = "fashionmnist_reduced"

            red_X_train, y_train, red_X_test, y_test = load_dataset_from_file(filename=filename, num_components=self.num_components)
        
        self.red_X_train = red_X_train
        self.red_X_test  = red_X_test
        self.y_train = y_train
        self.y_test = y_test
    
    def fit(self, X=None, y=None, time_t=True):
        """
        Sequentially transform the data and fit the transformed data using the 
        final estimator.
        
        X - Training data
        y - Training targets
        time_t - True => output times
        """

        X = self.red_X_train
        y = self.y_train
        
        start_time = time.perf_counter()
        self.SVC = self.SVC.fit(X=X, y=y)
        end_time = time.perf_counter()

        # if (time_t):
        #     print(f"elapsed time: {end_time - start_time}")

        #SVC is now fitted
        return end_time - start_time


    def predict(self, X = None):
        """
        Transform the data, and apply predict with the final estimator.
        
        X - Data to predict on
        """

        if (X is None):
            X = self.red_X_test

        return self.SVC.predict(X)
