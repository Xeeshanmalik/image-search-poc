try:
    from sklearn.decomposition import PCA

    # More than One Dimensionality Reduction Techniques can be used here to improve Accuracy

    class Reduction:

        def __do_pca_model_1__(self):

            pca = PCA(n_components=1000)
            pca.fit(self.features)
            pca_features = pca.transform(self.features)
            return pca_features

        def __init__(self, features):

            self.features = features

except ImportError as E:
    raise E