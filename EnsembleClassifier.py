import numpy as np
from sklearn.svm import SVC 


class EnsembleClassifier:
    def __init__(self, svc_type="linear", params=(10, None, None)):
        self.classifiers = []
        self.max_clfs = 8
        self.svc_type = svc_type

        if self.svc_type == "linear" and params[1:] != (None, None):
            raise ValueError(f"Linear kernel only uses C, do not need extra params: {params[1:]}")

    def fit(self, training_data, num_subsets, kernel_type):
        print(training_data.shape)

        # generate num_subsets training samples by sampling training_data
        for i in range(num_subsets):
            row_indices = np.random.choice(len(training_data), size=len(training_data), replace=True)
            sampled_data = training_data[row_indices]

            X_train = sampled_data[:, :-1]
            y_train = sampled_data[:, -1]


    def predict(self, X):
        pass
        


m = EnsembleClassifier()

from test import test


m.fit(test(), 4)
