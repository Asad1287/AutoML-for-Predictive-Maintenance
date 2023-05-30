from src.Data_Processing.dim_reduction import * 

import dask
from typing import List
from dask.distributed import Client
import joblib 
import os 
import pandas as pd
import dask.dataframe as dd

def dim_reduction_processing(col_names:List[str],parquet_file:str,save_parquet_file_path:str) -> None:
    #col_names =['UDI', 'Product ID', 'Type', 'Air temperature',
    #    'Process temperature', 'Rotational speed', 'Torque',
    #    'Tool wear', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    #FEATURE_STORE = "/mnt/d/Portfolio/Auto_ML_Pdm/AutoML/src/FeatureStore"
    #parquet_file = os.path.join(FEATURE_STORE, "processed_data.parquet")
    #save_parquet_file_path = os.path.join(FEATURE_STORE, "processed_data_reduced.parquet")
    #read from parquet file
    X_train = dd.read_parquet(parquet_file,engine="pyarrow")
    #load pickle y_train



    

    X_train.columns = col_names
    #X_test.columns = col_names

    #X_test['Target'] = y_test
    
    #X_train = X_train.dropna()
    #X_test = X_test.dropna()

    #y_test= X_test.pop("Target")

    dim_reducer = DimensionalityReducer(3)
    dim_reducer.fit(X_train)
    X_train_reduced = dim_reducer.transform(X_train)
    print(X_train_reduced.shape)
    print(X_train_reduced.head())
    X_train_reduced.to_parquet(save_parquet_file_path,engine="pyarrow")
