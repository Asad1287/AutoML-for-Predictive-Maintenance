import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from typing import List
class Feature_Engineering:
    def __init__(self, dataframe:pd.DataFrame):
        self.dataframe = dataframe
    
    def apply_polynomial_features(self, apply_cols:List[str],  degree:int=2, interaction_only:bool=False, include_bias:bool=True):
        #apply polynomial features to dataframe and add to dataframe
        poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias)
        poly_features = poly.fit_transform(self.dataframe[apply_cols])
        #replace the original columns with the new polynomial features
        self.dataframe = self.dataframe.drop(apply_cols, axis=1)
        self.dataframe = pd.concat([self.dataframe, pd.DataFrame(poly_features)], axis=1)
        return self.dataframe
    def interaction_features(self, apply_cols:List[str]):
        #create interaction features
        new_feature = None 
        for feature in apply_cols:
            if new_feature is None:
                new_feature = self.dataframe[feature]
            else:
                new_feature = new_feature * self.dataframe[feature]
        self.dataframe = self.dataframe.drop(apply_cols, axis=1)
        self.dataframe = pd.concat([self.dataframe, pd.DataFrame(new_feature)], axis=1)
        return self.dataframe
    def ratio_features(self, apply_cols:List[str]):
        #create ratio features
        pass 
    def log_features(self, apply_cols:List[str]):
        #create log features
        pass

    def binned_features(self, num_col:str, n_bins:int):
        def bin_continuous_to_categorical(series, n_bins):
            # Calculate the bin edges for the histogram
            bin_edges = np.histogram_bin_edges(series, bins=n_bins)

            # Bin the continuous variable with pandas.cut and convert it to a categorical variable
            categorical = pd.cut(series, bins=bin_edges, include_lowest=True, labels=False)

            return categorical
        self.dataframe[num_col] = bin_continuous_to_categorical(self.dataframe[num_col], n_bins)
        return self.dataframe
    def one_hot_encode(self, apply_cols:List[str]):
        #create one hot encoded features
        self.dataframe = pd.get_dummies(self.dataframe, columns=apply_cols)
        return self.dataframe
    