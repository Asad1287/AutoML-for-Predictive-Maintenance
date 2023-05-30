import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class Correlation_Analysis(BaseEstimator, TransformerMixin):
    def __init__(self, target:str, threshold:int=0.5):
        self.target = target
        self.threshold = threshold

    def fit(self, X):
        self.correlations = X.corr()[self.target]
        self.high_correlation_features = self.correlations[abs(self.correlations) > self.threshold].index
        return self

    def transform(self, X):
        return X[self.high_correlation_features]

    def plot_heatmap(self, X):
        plt.figure(figsize=(12, 10))
        sns.heatmap(X[self.high_correlation_features].corr(), annot=True, fmt=".2f")
        plt.show()

import unittest

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class TestCorrelationAnalysis(unittest.TestCase):
    def setUp(self):
        iris = datasets.load_iris()
        self.X = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])
        self.transformer = Correlation_Analysis(target='target')

    def test_fit(self):
        # Ensure it returns self
        self.assertEqual(self.transformer.fit(self.X), self.transformer)

    def test_transform(self):
        self.transformer.fit(self.X)
        # Ensure that the returned DataFrame only contains high correlation features
        transformed = self.transformer.transform(self.X)
        for feature in transformed.columns:
            self.assertIn(feature, self.transformer.high_correlation_features)

    def test_plot_heatmap(self):
        # This is a bit tricky to test as it's a plot, so just check it runs
        try:
            self.transformer.fit(self.X)
            self.transformer.plot_heatmap(self.X)
        except Exception as e:
            self.fail(f"plot_heatmap raised {type(e).__name__} unexpectedly!")

if __name__ == '__main__':
    unittest.main()