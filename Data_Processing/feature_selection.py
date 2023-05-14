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
