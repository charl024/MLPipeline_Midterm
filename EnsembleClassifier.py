import numpy as np
from sklearn.svm import SVC 


class EnsembleClassifier:
    def __init__(self, svc_type="linear", params=(10, None, None), max_clfs=8, max_iter=50, eta=0.001):
        self.classifiers = []
        self.max_clfs = max_clfs
        self.svc_type = svc_type
        self.params = params
        self.max_iter = max_iter
        self.eta = eta

        if self.svc_type == "linear" and self.params[1:] != (None, None):
            raise ValueError(f"Linear kernel only uses C, do not need extra params: {self.params[1:]}")

        if self.svc_type == "rbf" and self.params[2] is not None:
            raise ValueError(f"RBF kernel only uses C and gamma, no degree needed: {self.params[2]}")

        if self.svc_type == "poly" and any(p is None for p in self.params):
            raise ValueError(f"Poly kernel requires all params (C, gamma, degree), got: {self.params}")

    def fit(self, X, y):

        # generate num_subsets training samples by sampling X
        for i in range(self.max_clfs):
            row_indices = np.random.choice(len(X), size=len(X), replace=True)

            X_sample = X[row_indices]
            y_sample = y[row_indices]

            svc = self.choose_svc()
            svc.fit(X=X_sample, y=y_sample)

            self.classifiers.append(svc)


    def predict(self, X):
        all_preds = np.array([clf.predict(X) for clf in self.classifiers])

        majority_vote = []

        for sample_preds in all_preds.T:
            vote = np.bincount(sample_preds.astype(int)).argmax()
            majority_vote.append(vote)
        return np.array(majority_vote)

    def choose_svc(self):
        match self.svc_type:
            case "linear":
                return SVC(kernel='linear', C=self.params[0], max_iter=self.max_iter, tol=self.eta)
            case "rbf":
                return SVC(kernel='rbf', C=self.params[0], gamma=self.params[1], max_iter=self.max_iter, tol=self.eta)
            case "poly":
                return SVC(kernel='poly', C=self.params[0], gamma=self.params[1], degree=self.params[2], max_iter=self.max_iter, tol=self.eta)
            case _:
                print("Return default SVC(Linear)")
                return SVC(kernel='linear', C=self.params[0], max_iter=self.max_iter, tol=self.eta)