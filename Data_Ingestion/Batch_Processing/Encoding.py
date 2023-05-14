from category_encoders import TargetEncoder, OrdinalEncoder, BinaryEncoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
import pandas as pd
import numpy as np
from typing import List


class Encoding:
    def __init__(self, df_train:pd.DataFrame, df_test:pd.DataFrame, target_col:str, feature_cols:List[str]):
        self.df_train = df_train
        self.df_test = df_test
        self.target_col = target_col
        self.feature_cols = feature_cols

    def label_encoding(self):
        le = LabelEncoder()
        df_train_encoded = self.df_train.copy()
        df_test_encoded = self.df_test.copy()
        for col in self.feature_cols:
            df_train_encoded[col] = le.fit_transform(df_train_encoded[col])
            df_test_encoded[col] = df_test_encoded[col].map(lambda s: 'unknown' if s not in le.classes_ else s)
            le.classes_ = np.append(le.classes_, 'unknown')
            df_test_encoded[col] = le.transform(df_test_encoded[col])
        return df_train_encoded, df_test_encoded
    
    def label_encoding_dask(self):
        for col in self.feature_cols:
            # compute unique values from train set
            unique_values_train = self.df_train[col].unique().compute()
            # create mapping dictionary from unique values in train set
            mapping = {value: i for i, value in enumerate(unique_values_train)}
            # add one additional category for unknown values
            mapping['unknown'] = len(mapping)
            # apply mapping to train set
            self.df_train[col] = self.df_train[col].map(mapping).fillna(mapping['unknown'])
            # apply mapping to test set, new unknown values will be filled with 'unknown' mapping
            self.df_test[col] = self.df_test[col].map(mapping).fillna(mapping['unknown'])
        return self.df_train, self.df_test

    def one_hot_encoding(self):
        ohe = OneHotEncoder(drop='first', handle_unknown='ignore')
        df_train_encoded = self.df_train.copy()
        df_test_encoded = self.df_test.copy()
        df_train_encoded = pd.get_dummies(df_train_encoded, columns=self.feature_cols)
        df_test_encoded = pd.get_dummies(df_test_encoded, columns=self.feature_cols)
        return df_train_encoded, df_test_encoded

    def target_encoding(self):
        te = TargetEncoder()
        df_train_encoded = self.df_train.copy()
        df_test_encoded = self.df_test.copy()
        df_train_encoded[self.feature_cols] = te.fit_transform(df_train_encoded[self.feature_cols], df_train_encoded[self.target_col])
        df_test_encoded[self.feature_cols] = te.transform(df_test_encoded[self.feature_cols])
        return df_train_encoded, df_test_encoded

    def ordinal_encoding(self):
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
        df_train_encoded = self.df_train.copy()
        df_test_encoded = self.df_test.copy()
        df_train_encoded[self.feature_cols] = oe.fit_transform(df_train_encoded[self.feature_cols])
        df_test_encoded[self.feature_cols] = oe.transform(df_test_encoded[self.feature_cols])
        return df_train_encoded, df_test_encoded

    def binary_encoding(self):
        be = BinaryEncoder(handle_unknown='ignore')
        df_train_encoded = self.df_train.copy()
        df_test_encoded = self.df_test.copy()
        df_train_encoded = be.fit_transform(df_train_encoded[self.feature_cols])
        df_test_encoded = be.transform(df_test_encoded[self.feature_cols])
        return df_train_encoded, df_test_encoded

    def best_encoding(self):
        encoding_methods = {
            'Label Encoding': self.label_encoding(),
            'One-Hot Encoding': self.one_hot_encoding(),
            'Target Encoding': self.target_encoding(),
            'Ordinal Encoding': self.ordinal_encoding(),
            'Binary Encoding': self.binary_encoding(),
        }
        
        best_score = -np.inf
        best_method = None
        
        for method, (df_train_encoded, df_test_encoded) in encoding_methods.items():
            X_train = df_train_encoded[self.feature_cols]
            y_train = df_train_encoded[self.target_col]
            X_test = df_test_encoded[self.feature_cols]
            y_test = df_test_encoded[self.target_col]
            
            rf = RandomForest