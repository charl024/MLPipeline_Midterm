from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# input train/test datasets, with specified num components (reduced dimensionality)
# return reduced datasets
def get_samples_with_pca(train_data, test_data, n_components):
    '''
    Perform dimensionality reduction using Principal Component Analysis (PCA).

    PCA finds principal components that maximize variance in the data.

    Parameters
    ----------
    train_data : array-like of shape (n_samples, n_features)
    test_data : array-like of shape (n_samples, n_features)
    n_components : int
        Number of principal components to keep.

    Returns
    -------
    reduced_train : ndarray of shape (n_samples, n_components)
    reduced_test : ndarray of shape (n_samples, n_components)
    -------
    '''
    pca = PCA(n_components=n_components)
    reduced_train = pca.fit_transform(train_data)
    reduced_test = pca.transform(test_data)
    return reduced_train, reduced_test

def get_samples_with_lda(train_data, train_labels, test_data):
    '''
    Perform dimensionality reduction using Linear Discriminant Analysis (LDA).

    Parameters
    ----------
    train_data : array-like of shape (n_samples, n_features)
    train_labels : array-like of shape (n_samples,)
    test_data : array-like of shape (n_samples, n_features)

    Returns
    -------
    reduced_train : ndarray of shape (n_samples, n_components)
    reduced_test : ndarray of shape (n_samples, n_components)
    -------
    '''
    lda = LDA()
    reduced_train = lda.fit_transform(train_data, train_labels)
    reduced_test = lda.transform(test_data)
    return reduced_train, reduced_test