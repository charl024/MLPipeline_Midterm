"""
A pipeline class for testing various models
"""
from dataset_reduction import load_dataset_from_file
from sklearn.svm import SVC 
import import_datasets
import pca_lda

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
        if data_set == MNST_STR:
            self.data_set = import_datasets.get_mnist
        else: 
            self.data_set = import_datasets.get_fashion_mnist

        #Define the preprocessing strategy
        if preprocessor == "pca":
            self.preprocessor = pca_lda.get_samples_with_pca
            self.num_components=num_components
        else:
            self.preprocessor = pca_lda.get_samples_with_lda
            self.num_components = None

        # SVC
        self.SVC = SVC
        
        # Initialize unpreprocesed data
        self._load_uncompressed()

        # Load in preprocessed data
        self._transform()

    def _load_uncompressed(self):
        """
        Load the data
        """
        X_train, y_train, X_test, y_test = self.data_set()

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

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

            red_X_train, _, red_X_test, _ = load_dataset_from_file(filename=filename, num_components=self.num_components)

        except ValueError as e:
            print(f"An exception occurred: {e}")
            
        #TODO: file reading instead of function calls every time

        if (red_X_test == None or red_X_test == None):
            "failed to read in, doing manually"
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
        
        return self.X_train, self.y_train, self.X_test, self.y_test, self.red_X_train, self.red_X_test
    
    def fit(self, X=None, y=None):
        """
        Sequentially transform the data and fit the transformed data using the 
        final estimator.
        
        X - Training data
        y - Training targets
        """

        X = self.red_X_train
        y = self.y_train
        
        self.SVC.fit(X=X, y=y)

        #SVC is now fitted
        return self


    def predict(self, X = None):
        """
        Transform the data, and apply predict with the final estimator.
        
        X - Data to predict on
        """

        X = self.red_X_test
        
        return self.SVC.predict(X)
