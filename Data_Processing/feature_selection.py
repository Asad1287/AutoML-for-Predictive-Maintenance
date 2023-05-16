from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import numpy as np

class FeatureSelector:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def select_features_random_forest(self, n_features):
        # Fit the model
        model = RandomForestClassifier()
        model.fit(self.X, self.y)
        # Get feature importances
        importances = model.feature_importances_
        # Get the indices of the top n features
        indices = np.argsort(importances)[::-1][:n_features]
        # Return the names of the top n features
        return self.X.columns[indices]

    def select_features_rfe(self, n_features):
        # Initialize the model
        model = LogisticRegression()
        # Initialize RFE
        rfe = RFE(estimator=model, n_features_to_select=n_features)
        # Fit RFE
        rfe.fit(self.X, self.y)
        # Get the feature ranking
        ranking = rfe.ranking_
        # Get the indices of the top n features
        indices = np.where(ranking==1)[0]
        # Return the names of the top n features
        return self.X.columns[indices]

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

class FeatureSelectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, method='random_forest', n_features=10):
        self.method = method
        self.n_features = n_features

    def fit(self, X, y=None):
        if self.method == 'random_forest':
            self.model = RandomForestClassifier()
        elif self.method == 'rfe':
            self.model = RFE(estimator=LogisticRegression(), n_features_to_select=self.n_features)
        self.model.fit(X, y)
        return self

    def transform(self, X):
        if self.method == 'random_forest':
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1][:self.n_features]
        elif self.method == 'rfe':
            ranking = self.model.ranking_
            indices = np.where(ranking==1)[0]
        return X.iloc[:, indices]

import unittest
from sklearn.datasets import load_iris
import pandas as pd

class TestFeatureSelector(unittest.TestCase):
    def setUp(self):
        iris = load_iris()
        self.X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        self.y = iris.target
        self.fs = FeatureSelector(self.X, self.y)

    def test_select_features_random_forest(self):
        selected_features = self.fs.select_features_random_forest(2)
        self.assertEqual(len(selected_features), 2)
        for feature in selected_features:
            self.assertTrue(feature in self.X.columns)

    def test_select_features_rfe(self):
        selected_features = self.fs.select_features_rfe(2)
        self.assertEqual(len(selected_features), 2)
        for feature in selected_features:
            self.assertTrue(feature in self.X.columns)

if __name__ == '__main__':
    unittest.main()
