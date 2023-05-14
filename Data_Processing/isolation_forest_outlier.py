import unittest 
import pandas as pd 
from sklearn.ensemble import IsolationForest
class IsolationForestOutlierDetector:
    def __init__(self, data: pd.DataFrame, contamination=0.1):
        if isinstance(data, pd.DataFrame):
            self.data = data.values
        else:
            self.data = data
        self.detector = IsolationForest(contamination=contamination)
    
    def fit(self):
        self.detector.fit(self.data)
    
    def predict(self,remove=False):
        outliers = self.detector.predict(self.data)
        if remove:
            return self.data[outliers == 1]
        else:
            outliers = self.data[outliers == -1]
            return outliers if outliers.size > 0 else None

import unittest
import pandas as pd


class TestIsolationForestOutlierDetector(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({'A': [1,1,1,1,1,1,1,1,10]})
        self.detector = IsolationForestOutlierDetector(self.data)

    def test_predict(self):
        self.detector.fit()
        outliers = self.detector.predict()
        self.assertEqual(outliers[0][0], 10)
        

if __name__ == "__main__":
    unittest.main()