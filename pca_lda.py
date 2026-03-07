from sklearn.decomposition import PCA

# input train/test datasets, with specified num components (reduced dimensionality)
# return reduced datasets
def get_samples_with_pca(train_data, test_data, n_components):
    pca = PCA(n_components=n_components)
    reduced_train = pca.fit_transform(train_data)
    reduced_test = pca.transform(test_data)
    return reduced_train, reduced_test