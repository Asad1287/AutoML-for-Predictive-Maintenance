from src.Data_Processing.basic_ft_eng_process import * 
from src.Data_Processing.correlation import *
import dask
from typing import List
from dask.distributed import Client
import joblib 
import os 
import pandas as pd
import dask.dataframe as dd

def correlation_processing(target_path:str,col_names:List[str],source_parquet_file:str,dest_parquet_file_path:str) -> None:
    #col_names =['UDI', 'Product ID', 'Type', 'Air temperature',
    #    'Process temperature', 'Rotational speed', 'Torque',
    #    'Tool wear', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    
        #read from parquet file
    X_train = dd.read_parquet(source_parquet_file,engine="pyarrow")
        #load pickle y_train


    
        

    X_train.columns = col_names
    y_train = joblib.load(target_path)

    corr = Correlation_Analysis(y_train)
    corr.fit(X_train)
    X_train_corr = corr.transform(X_train)
    print(X_train_corr.shape)
FEATURE_STORE = "/mnt/d/Portfolio/Auto_ML_Pdm/AutoML/src/FeatureStore"
target_path = "y_train.pkl"
col_names =['UDI', 'Product ID', 'Type', 'Air temperature',
        'Process temperature', 'Rotational speed', 'Torque',
        'Tool wear', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
source_parquet_file = os.path.join(FEATURE_STORE, "processed_data.parquet")
dest_parquet_file_path = os.path.join(FEATURE_STORE, "corr_processed_data_reduced.parquet")
correlation_processing(target_path,col_names,source_parquet_file,dest_parquet_file_path)

