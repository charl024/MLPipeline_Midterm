import numpy as np


class EnsembleClassifier:
    def __init__(self):
        self.classifiers = []

    def fit(self, training_data, num_subsets, kernel_type):
        print(training_data.shape)

        # generate num_subsets training samples by sampling training_data
        for i in range(num_subsets):
            row_indices = np.random.choice(len(training_data), size=len(training_data), replace=True)
            sampled_data = training_data[row_indices]

            # train SVC w kernel_type

    def predict(self, X):
        pass
        


m = EnsembleClassifier()

from test import test


m.fit(test(), 4)
