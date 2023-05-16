from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import dask.dataframe as dd
from typing import Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from typing import Tuple

class DimensionalityReducer(BaseEstimator, TransformerMixin):
    def __init__(self, target:str, n_components:int):
        self.target = target
        self.n_components = n_components

    def fit(self, X: pd.DataFrame, y = None):
        # The target column is not used to fit the transformer in PCA,
        # so we only store it to use later in the transform method
        self.target_ = X[self.target]
        self.pca_ = PCA(n_components=self.n_components)
        self.pca_.fit(X.drop(self.target, axis=1))
        return self

    def transform(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        X_transformed = self.pca_.transform(X.drop(self.target, axis=1))
        # Return the transformed features along with the target column
        return pd.DataFrame(X_transformed), self.target_

import unittest
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

class TestDimensionalityReducer(unittest.TestCase):
    def setUp(self):
        self.iris = load_iris()
        self.df = pd.DataFrame(data= np.c_[self.iris['data'], self.iris['target']],
                     columns= self.iris['feature_names'] + ['target'])
        self.reducer = DimensionalityReducer(target='target', n_components=2)

    def test_fit_transform(self):
        X_transformed, y = self.reducer.fit_transform(self.df)
        self.assertEqual(X_transformed.shape[1], 2)  # check if number of components is 2
        self.assertEqual(X_transformed.shape[0], self.df.shape[0])  # check if number of samples is preserved
        self.assertTrue(isinstance(X_transformed, pd.DataFrame))  # check if it returns DataFrame
        self.assertTrue(isinstance(y, pd.Series))  # check if it returns Series
        self.assertTrue((y == self.df['target']).all())  # check if target values are preserved

if __name__ == "__main__":
    unittest.main()