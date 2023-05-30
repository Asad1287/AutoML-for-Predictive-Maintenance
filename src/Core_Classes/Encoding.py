import pandas as pd
import numpy as np
from typing import List
from sklearn.preprocessing import LabelEncoder
import dask.dataframe as dd
import joblib

class Encoding:
    def __init__(self, df_train: dd.DataFrame,  feature_cols:List[str]):
        self.df_train = df_train
        
        self.feature_cols = feature_cols
        self.encodings = {}

    def label_encoding_fit(self):
        for col in self.feature_cols:
            # compute unique values from train set
            unique_values_train = self.df_train[col].unique().compute()
            # create mapping dictionary from unique values in train set
            mapping = {value: i for i, value in enumerate(unique_values_train)}
            # add one additional category for unknown values
            mapping['unknown'] = len(mapping)
            # store the encoding in the encodings dictionary
            self.encodings[col] = mapping
            # apply mapping to train set
            self.df_train[col] = self.df_train[col].map(mapping).fillna(mapping['unknown'])

        joblib.dump(self.encodings, 'encodings.joblib')  # save the encodings
        return self.df_train

    def label_encoding_transform(self, df_test: dd.DataFrame):
        self.encodings = joblib.load('encodings.joblib')  # load the saved encodings

        for col in self.feature_cols:
            # load mapping from encodings
            mapping = self.encodings.get(col)
            # apply mapping to test set, new unknown values will be filled with 'unknown' mapping
            df_test[col] = df_test[col].map(mapping).fillna(mapping['unknown'])
        return df_test
