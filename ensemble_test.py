from EnsembleClassifier import EnsembleClassifier
from dataset_reduction import load_dataset_from_file
import numpy as np
import time


NUM_ITERATIONS = 512

NUM_COMPS = 50

# For linear: C = 0.01
LINEAR_PARAMS = (0.01, None, None)

# For rbf, 8192: C = 10, gamma = 0.001
#RBF_PARAMS = (10, 0.001, None)
# For rbf, 512:C = 1, gamma = 0.01
RBF_PARAMS = (1, 0.01, None)

# For poly: C = 0.1, gamma = 0.01, degree = 3 
POLY_PARAMS = (0.1, 0.01, 3)

def get_accuracy(preds, actuals):
    """
    Get the proportion of correct predictions

    preds   - predicted labels
    actuals - true labels
    """
    return np.mean(preds == actuals)

def run_test(ensm, num_comps):
    # Load data
    X_train, y_train, X_test, y_test = load_dataset_from_file("mnist_reduced", num_components=num_comps)

    # Learn machine
    start_time = time.perf_counter()
    ensm.fit(X_train, y_train)
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time

    # Get accuracy
    train_acc = get_accuracy(ensm.predict(X_train), y_train)
    test_acc = get_accuracy(ensm.predict(X_test), y_test)

    print(f"training accuracy = {train_acc}, test accuracy = {test_acc}, fit time = {elapsed_time}\n")

    return (train_acc, test_acc, elapsed_time)


def test_linear_ensemble(num_comps=NUM_COMPS):
    """Linear ensemble classifier test"""

    print(f"Number of Iterations = {NUM_ITERATIONS}")
    print(f"Number of Components for PCA reduction = {num_comps}")
    print(f"C = {LINEAR_PARAMS[0]}")

    # Ensemble Classifier
    ensm = EnsembleClassifier(svc_type="linear", params=LINEAR_PARAMS, max_iter=NUM_ITERATIONS)
    run_test(ensm=ensm, num_comps=num_comps)

def test_rbf_ensemble(num_comps=NUM_COMPS):
    """RBF ensemble classifier test"""

    print(f"Number of Iterations = {NUM_ITERATIONS}")
    print(f"Number of Components for PCA reduction = {num_comps}")
    print(f"C = {RBF_PARAMS[0]}, gamma = {RBF_PARAMS[1]}")

    # Ensemble Classifier
    ensm = EnsembleClassifier(svc_type="rbf", params=RBF_PARAMS, max_iter=NUM_ITERATIONS)
    run_test(ensm=ensm, num_comps=num_comps)

def test_poly_ensemble(num_comps=NUM_COMPS):
    """Poly ensemble classifier test"""

    print(f"Number of Iterations = {NUM_ITERATIONS}")
    print(f"Number of Components for PCA reduction = {num_comps}")
    print(f"C = {POLY_PARAMS[0]}, gamma = {POLY_PARAMS[1]}, degree = {POLY_PARAMS[2]}")

    # Ensemble Classifier
    ensm = EnsembleClassifier(svc_type="poly", params=POLY_PARAMS, max_iter=NUM_ITERATIONS)
    run_test(ensm=ensm, num_comps=num_comps)

if __name__ == "__main__":
    component_number = [50, 100, 200]
    for n in component_number:
        test_linear_ensemble(n)