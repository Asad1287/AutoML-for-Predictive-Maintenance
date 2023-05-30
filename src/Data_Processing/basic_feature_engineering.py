from src.Data_Processing.basic_ft_eng_process import * 

import dask
from typing import List
from dask.distributed import Client
import joblib 
import os 
import pandas as pd
import dask.dataframe as dd

def basic_eng_processing(col_names:List[str],parquet_file:str,save_parquet_file_path:str) -> None:
    #col_names =['UDI', 'Product ID', 'Type', 'Air temperature',
    #    'Process temperature', 'Rotational speed', 'Torque',
    #    'Tool wear', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    
        #read from parquet file
    X_train = dd.read_parquet(parquet_file,engine="pyarrow")
        #load pickle y_train


    col_names =['UDI', 'Product ID', 'Type', 'Air temperature',
        'Process temperature', 'Rotational speed', 'Torque',
        'Tool wear', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        

    X_train.columns = col_names
        #X_test.columns = col_names

        #X_test['Target'] = y_test
        
        #X_train = X_train.dropna()
        #X_test = X_test.dropna()

        #y_test= X_test.pop("Target")

    basic_eng = BasicFeatureEngineering(X_train)

    processed_df = basic_eng.multiply_columns("Rotational speed","Torque","Torque_Rotational_speed")

    
    print(processed_df.head())
    processed_df.to_parquet(save_parquet_file_path,engine="pyarrow")
