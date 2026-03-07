"""
A pipeline class for testing various models
"""
from pca_lda import get_samples_with_pca
from sklearn.svm import SVC 
import import_datasets
import pca_lda


class Pipeline:
    def __init__(self,
                 data_set="mnist",
                 preprocessor="pca", 
                 SVC=SVC(max_iter=10),
                 num_components=50):
        """
        A Pipeline object
        preprocessor   - "pca" for PCA, LDA otw
        num_components - only relevant for PCA preprocessing
        X_test         - the test data
        """
        

        #Define the preprocessing strategy
        if preprocessor == "pca":
            self.preprocessor = pca_lda.get_samples_with_pca
            self.num_components=num_components
        else:
            self.preprocessor = pca_lda.get_samples_with_lda
            self.num_components = None

        # SVC
        self.SVC = SVC
        
        # Test data
        self._init(data_set=data_set)

    def _init(self, data_set="mnist"):
        """
        Load the data

        data_set- "mnist" for mnist, fahsion otw
        """
        #Define function for reading in the datasets
        if data_set == "mnist":
            getter = import_datasets.get_mnist
        else:
            getter = import_datasets.get_fashion_mnist
        
        X_train, y_train, X_test, y_test = getter()

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def _transform(self):
        """
        Perform preprocessing on X_train and X_test
        """
        #TODO: file reading instead of function calls every time
        if (self.num_components == None):
            # LDA
            red_X_train, red_X_test = self.preprocessor(self.X_train, 
                                                        self.y_train, 
                                                        self.X_test)
        else:
            # PCA
            red_X_train, red_X_test = self.preprocessor(self.X_train, 
                                                        self.X_test, 
                                                        self.num_components)
        
        self.red_X_train = red_X_train
        self.red_X_test  = red_X_test

    def get_data(self):
        """
        Get the data, not preprocessed
        """
        return self.X_train, self.y_train, self.X_test, self.y_test
    
    def fit(self, X=None, y=None):
        """
        Sequentially transform the data and fit the transformed data using the 
        final estimator.
        
        X - Training data
        y - Training targets
        """
        if ((X == None and y != None) or (X != None and y == None)):
            print(f"Warning, unexpected input combination in Pipeline: \nX = {X}\ny = {y}")
        if (X == None):
            X = self.red_X_train
        if (y == None):
            y = self.y_train
        
        self.SVC.fit(X=X, y=y)

        #SVC is now fitted
        return self


    def predict(self, X = None):
        """
        Transform the data, and apply predict with the final estimator.
        
        X - Data to predict on
        """

        if (X == None):
            X = self.red_X_test
        
        return self.SVC.predict(X)
