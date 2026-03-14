from EnsembleClassifier import EnsembleClassifier
from dataset_reduction import load_dataset_from_file
import numpy as np

X_train, y_train, X_test, y_test = load_dataset_from_file("mnist_reduced", num_components=50)

Cs = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

for C in Cs:
    ensm = EnsembleClassifier(svc_type="linear", params=(C, None, None), max_iter=100)

    ensm.fit(X_train, y_train)

    preds = ensm.predict(X_test)

    accuracy = np.mean(preds == y_test)
    train_preds = ensm.predict(X_train)
    train_acc = np.mean(train_preds == y_train)

    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {accuracy:.4f}")