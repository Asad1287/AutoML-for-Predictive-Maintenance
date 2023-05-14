from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class DimensionalityReducer:
    def __init__(self, dataframe, target, n_components):
        self.dataframe = dataframe.drop(target, axis=1)
        self.target = dataframe[target]
        self.n_components = n_components

    def pca(self):
        pca = PCA(n_components=self.n_components)
        transformed = pca.fit_transform(self.dataframe)
        return transformed, self.target

    def tsne(self):
        tsne = TSNE(n_components=self.n_components)
        transformed = tsne.fit_transform(self.dataframe)
        return transformed, self.target

# Usage example:
# dr = DimensionalityReducer(df, 'target', 2)
# pca_data, target = dr.pca()
# tsne_data, target = dr.tsne()
