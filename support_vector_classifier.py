from sklearn.svm import SVC 
from sklearn.model_selection import validation_curve

"""
The following file is used for comparing the differrence between the three types
of kernels supported by Scikit-learn's SVC class:

linear - Linear kernel, hyperparameter: C
rbf    - Radial basis function kernel, hyperparameters: C and gamma.
poly   - Polynomial kernel, hyperparametes: C, gamma, and degree.
"""

# Maximum number of itterations for training each SVC
MAX_ITER = 10

Cs      = [0.1, 1] # All three
gammas  = [0.1, 1] # rbf + poly
degrees = [0.1, 1] # Only poly

def gen_SVCs():
    """
    Generate the SVC's for each kernel, given all possible combinations of hyper
    parameters
    RETURN: the SVCs
    """

    # An array of the Support Vector Classifiers
    linears = []
    rbfs    = []
    polys   = []

    for C in Cs:
        # Generate the linear SVC's
        linears.append(SVC(kernel='linear',
                           C=C,
                           max_iter=MAX_ITER))
        for gamma in gammas:
            # Generate the rbfs
            rbfs.append(SVC(kernel='rbf',
                            C=C, 
                            gamma=gamma,
                            max_iter=MAX_ITER))
            for degree in degrees:
                polys.append(SVC(kernel='poly',
                                 C=C, 
                                 gamma=gamma, 
                                 degree=degree,
                                 max_iter=MAX_ITER))
    return linears, rbfs, polys
