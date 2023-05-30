from sklearn.impute import SimpleImputer
from dask_ml.impute import SimpleImputer as DaskSimpleImputer
from dask_ml.preprocessing import Categorizer, DummyEncoder
import dask.dataframe as dd
import pandas as pd
import joblib
class Imputation:
    def __init__(self, df_train: dd.DataFrame):
        self.df_train = df_train
        self.imputer = None
        self.categorizer = None
    
    def fit(self, imputer_type='simple'):
        if imputer_type == 'simple':
            self.imputer = DaskSimpleImputer(strategy='most_frequent')
            self.imputer.fit(self.df_train)
        else:
            self.categorizer = Categorizer()
            self.categorizer.fit(self.df_train)
    
    def transform(self, df, imputer_type='simple'):
        if imputer_type == 'simple':
            df_imputed = self.imputer.transform(df)
        else:
            df_imputed = self.categorizer.transform(df)
        return df_imputed
    
    def fit_transform(self, imputer_type='simple'):
        self.fit(imputer_type)
        return self.transform(self.df_train, imputer_type)