import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.cluster import DBSCAN


class OutlierDetectorTransformer:
    def __init__(self, data):
        if isinstance(data, pd.DataFrame):
            self.data = data.values
        else:
            self.data = data

    def z_score(self, threshold=2.0, remove=False):
        z_scores = np.abs(zscore(self.data))
        outliers = np.where(z_scores > threshold)
        if remove:
            return np.delete(self.data, outliers)
        else:
            return self.data[outliers]
        
    def modified_z_score(self, threshold=3.5, remove=False):
        one_sd = 0.6745
        median = np.median(self.data)
        median_absolute_deviation = np.median([np.abs(y - median) for y in self.data])
        modified_z_scores = [one_sd * (y - median) / median_absolute_deviation for y in self.data]
        outliers = np.where(np.abs(modified_z_scores) > threshold)
        if remove:
            return np.delete(self.data, outliers)
        else:
            return self.data[outliers]

    def iqr(self, k=1.5, remove=False):
        q75, q25 = np.percentile(self.data, [75 ,25])
        iqr = q75 - q25
        outliers = np.where((self.data < q25 - k*iqr) | (self.data > q75 + k*iqr))
        if remove:
            return np.delete(self.data, outliers)
        else:
            return self.data[outliers]

    def dbscan(self, eps=0.5, min_samples=5, remove=False):
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(self.data.reshape(-1,1))
        outliers = np.where(db.labels_ == -1)
        if remove:
            return np.delete(self.data, outliers)
        else:
            return self.data[outliers]
        


class OutlierDetector:
    def __init__(self, data):
        if isinstance(data, pd.DataFrame):
            self.data = data.values
        else:
            self.data = data

    def z_score(self, threshold=2.0, remove=False):
        z_scores = np.abs(zscore(self.data))
        outliers = np.where(z_scores > threshold)
        if remove:
            return np.delete(self.data, outliers)
        else:
            return self.data[outliers]
        
    def modified_z_score(self, threshold=3.5, remove=False):
        one_sd = 0.6745
        median = np.median(self.data)
        median_absolute_deviation = np.median([np.abs(y - median) for y in self.data])
        modified_z_scores = [one_sd * (y - median) / median_absolute_deviation for y in self.data]
        outliers = np.where(np.abs(modified_z_scores) > threshold)
        if remove:
            return np.delete(self.data, outliers)
        else:
            return self.data[outliers]

    def iqr(self, k=1.5, remove=False):
        q75, q25 = np.percentile(self.data, [75 ,25])
        iqr = q75 - q25
        outliers = np.where((self.data < q25 - k*iqr) | (self.data > q75 + k*iqr))
        if remove:
            return np.delete(self.data, outliers)
        else:
            return self.data[outliers]

    def dbscan(self, eps=0.5, min_samples=5, remove=False):
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(self.data.reshape(-1,1))
        outliers = np.where(db.labels_ == -1)
        if remove:
            return np.delete(self.data, outliers)
        else:
            return self.data[outliers]

import numpy as np
detector = OutlierDetector(np.array([1,1,1,1,1,1,1,1,10]))
print(detector.modified_z_score())
import unittest
class TestOutlierDetector(unittest.TestCase):
    def test_z_score(self):
        self.assertEqual(detector.z_score().size, 1)
        self.assertEqual(detector.z_score(remove=True).size, 8)
        
    def test_modified_z_score(self):
        self.assertEqual(detector.modified_z_score().size, 1)
        self.assertEqual(detector.modified_z_score()[0], 10)
    def test_iqr(self):
        self.assertEqual(detector.iqr().size, 1)
        self.assertEqual(detector.iqr(remove=True).size, 8)
    def test_dbscan(self):
        self.assertEqual(detector.dbscan().size, 1)
        self.assertEqual(detector.dbscan(remove=True).size, 8)

if __name__ == '__main__':
    unittest.main()