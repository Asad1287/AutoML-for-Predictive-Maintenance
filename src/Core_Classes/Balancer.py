from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from typing import List
import dask.dataframe as dd

class Balancer:
    def __init__(self, df, target_col):
      
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

    def balance_data(self):
        counter = Counter(self.target)
        min_class = min(counter, key=counter.get)
        min_count = counter[min_class]
        
        
        
        over = SMOTE()
        under = RandomUnderSampler()
        steps = [('o', over), ('u', under)]
        pipeline = Pipeline(steps=steps)
        
        X, y = pipeline.fit_resample(self.features, self.target)
        balanced_df = pd.concat([pd.DataFrame(X, columns=self.features.columns), pd.Series(y, name=self.target_col)], axis=1)
        
        return balanced_df


class DaskOverSampler(TransformerMixin, BaseEstimator):
    def __init__(self, target_col, majority, minority):
        self.target_col = target_col
        self.majority = majority
        self.minority = minority
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Separate majority and minority classes
        df_majority = X[X[self.target_col]==self.majority]
        df_minority = X[X[self.target_col]==self.minority]

        # Count how many samples are in the majority class
        majority_count = len(df_majority)

        # Resample the minority class with replacement
        df_minority_oversampled = df_minority.sample(frac=majority_count/len(df_minority), replace=True)

        # Concatenate the majority class dataframe with the oversampled minority class dataframe
        df_oversampled = dd.concat([df_majority, df_minority_oversampled])
        
        # Shuffle the data
        df_oversampled = df_oversampled.sample(frac=1)
        
        return df_oversampled