import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter
from sklearn.utils.multiclass import type_of_target

class Balancer:
    def __init__(self, df, target_col):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame.")
        if not isinstance(target_col, str):
            raise ValueError("target_col must be a string.")
        if target_col not in df.columns:
            raise ValueError("target_col is not a column in df.")
        if type_of_target(df[target_col]) not in ['binary', 'multiclass']:
            raise ValueError("target_col must be binary or multiclass.")
        
        self.df = df
        self.target_col = target_col
        self.features = df.drop(target_col, axis=1)
        self.target = df[target_col]

    def check_balance(self):
        counter = Counter(self.target)
        for k,v in counter.items():
            pct = v / len(self.target) * 100
            print(f'Class={k}, n={v} ({pct}%)')
        return counter

    def balance_data(self, oversample_ratio=0.1, undersample_ratio=0.5):
        counter = Counter(self.target)
        min_class = min(counter, key=counter.get)
        min_count = counter[min_class]
        
        if oversample_ratio < min_count / len(self.target):
            oversample_ratio = min_count / len(self.target) + 0.1
            print(f"Oversample ratio was too low; setting to {oversample_ratio}")
        
        over = SMOTE(sampling_strategy=oversample_ratio)
        under = RandomUnderSampler(sampling_strategy=undersample_ratio)
        steps = [('o', over), ('u', under)]
        pipeline = Pipeline(steps=steps)
        
        X, y = pipeline.fit_resample(self.features, self.target)
        balanced_df = pd.concat([pd.DataFrame(X, columns=self.features.columns), pd.Series(y, name=self.target_col)], axis=1)
        
        return balanced_df

import unittest
import pandas as pd
from collections import Counter

class TestBalancer(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'target': [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
        })
        self.balancer = Balancer(self.df, 'target')

    def test_check_balance(self):
        counter = self.balancer.check_balance()
        self.assertEqual(counter[0], 7)
        self.assertEqual(counter[1], 3)

    def test_balance_data(self):
        balanced_df = self.balancer.balance_data()
        counter = Counter(balanced_df['target'])
        self.assertEqual(counter[0], 3) # 0.5 * original count of majority class
        self.assertEqual(counter[1], 4) # 0.1 * original count of majority class + original count of minority class

if __name__ == "__main__":
    unittest.main()