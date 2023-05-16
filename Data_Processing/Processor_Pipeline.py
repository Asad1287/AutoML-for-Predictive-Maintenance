import dask.dataframe as dd
import joblib
from sklearn.pipeline import Pipeline

from sklearn.ensemble import IsolationForest
from sklearn.base import BaseEstimator, TransformerMixin

class IsolationForestOutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, contamination=0.1):
        self.contamination = contamination

    def fit(self, X, y=None):
        self.iforest = IsolationForest(contamination=self.contamination)
        self.iforest.fit(X)
        return self

    def transform(self, X):
        preds = self.iforest.predict(X)
        return X[preds == 1]  # only keep inliers
    
FEATURE_STORE_PATH = "D:\Portfolio\Auto_ML_Pdm\AutoML\Data_Ingestion\Batch_Processing"

# Prepare the pipeline
pipeline = Pipeline([
    ('remove_outliers', IsolationForestOutlierRemover(contamination=0.1))
])
import os 

x_train_path = os.path.join(FEATURE_STORE_PATH, "x_train.pkl")
import pickle
X_train = pickle.load(open(x_train_path, "rb"))
