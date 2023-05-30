import pandas as pd
import numpy as np
from typing import List
import dask.dataframe as dd

def oversample_dask(df:dd.DataFrame, target_col:str,majority:str,minority:str) -> dd.DataFrame:
    # Separate majority and minority classes
    df_majority = df[df[target_col]==majority]
    df_minority = df[df[target_col]==minority]

    # Count how many samples are in the majority class
    majority_count = len(df_majority)

    # Resample the minority class with replacement
    df_minority_oversampled = df_minority.sample(frac=majority_count/len(df_minority), replace=True)

    # Concatenate the majority class dataframe with the oversampled minority class dataframe
    df_oversampled = dd.concat([df_majority, df_minority_oversampled])
    
    # Shuffle the data
    df_oversampled = df_oversampled.sample(frac=1)
    
    return df_oversampled