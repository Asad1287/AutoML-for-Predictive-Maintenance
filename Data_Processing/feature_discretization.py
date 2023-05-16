from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
import pandas as pd

class HistogramBinning(BaseEstimator, TransformerMixin):
    def __init__(self, n_bins=10, input_col=""):
        self.n_bins = n_bins
        self.input_col = input_col

    def fit(self, X, y=None):
        # Calculate the bin edges using numpy histogram_bin_edges
        _, bin_edges = np.histogram(X[self.input_col], bins=self.n_bins)
        self.bin_edges_ = bin_edges
        return self

    def transform(self, X):
        # Bin the continuous variable using pandas.cut
        X[self.input_col] = pd.cut(X[self.input_col], bins=self.bin_edges_, labels=False, include_lowest=True)
        return X
import unittest
from sklearn.datasets import load_iris

class TestHistogramBinning(unittest.TestCase):
    def setUp(self):
        self.iris = load_iris(as_frame=True)
        self.df = self.iris['data']

    def test_histogram_binning(self):
        transformer = HistogramBinning(n_bins=3, input_col='sepal length (cm)')
        transformed = transformer.fit_transform(self.df.copy())

        # Check if the transformed dataframe has the same shape as the original
        self.assertEqual(transformed.shape, self.df.shape)

        # Check if the transformed column values are between 0 and n_bins-1
        self.assertTrue((transformed['sepal length (cm)'] >= 0).all())
        self.assertTrue((transformed['sepal length (cm)'] < 3).all())

        # Check if the bin edges are correctly calculated and stored
        self.assertTrue(hasattr(transformer, 'bin_edges_'))
        self.assertEqual(len(transformer.bin_edges_), 4)  # n_bins + 1

if __name__ == '__main__':
    unittest.main()
