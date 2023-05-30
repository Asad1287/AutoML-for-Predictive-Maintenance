from sklearn.manifold import TSNE
import pandas as pd
import dask.dataframe as dd
from typing import Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from dask_ml.decomposition import PCA
from typing import Tuple


class DimensionalityReducer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components:int):
        self.n_components = n_components

    def fit(self, X: dd.DataFrame):
        self.pca_ = PCA(n_components=self.n_components)
        self.pca_.fit(X.to_dask_array(lengths=True))
        return self

    def transform(self, X: dd.DataFrame) -> dd.DataFrame:
        X_transformed = self.pca_.transform(X.to_dask_array(lengths=True))
        return dd.from_array(X_transformed, columns=[f'PC{i+1}' for i in range(self.n_components)])