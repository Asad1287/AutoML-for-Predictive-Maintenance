from pymongo import MongoClient
from dask import delayed
from dask.distributed import Client
import pandas as pd
from dask import compute
import dask.dataframe as dd
import os

PASSWORD = "1234"
INFERENCE_COLLECTION = "inference_results"
DATABASE = "predictive_maintenance"

col_names = ['UDI', 'Product ID', 'Type', 'Air temperature',
       'Process temperature', 'Rotational speed', 'Torque',
       'Tool wear', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF', 'Machine failure']
mongo_uri = f"mongodb+srv://root12345:{PASSWORD}@cluster1.b03tix4.mongodb.net/",f"{DATABASE}",f"{INFERENCE_COLLECTION}"
def insert_chunk(df, collection_name, uri, database):
    client = MongoClient(uri)
    records = df.to_dict('records')
    client[database][collection_name].insert_many(records)
    client.close()

@delayed
def dask_insert(df, collection_name, uri, database):
    df_grouped = df.groupby(df.index)
    results = [insert_chunk(group, collection_name, uri, database) for name, group in df_grouped]
    return results

def write_dask_dataframe_to_mongodb(df, collection_name, database, uri):
    dask_insert(df, collection_name, uri, database).compute()

if __name__ == "__main__":
    client = Client()
    
    

    # Load data from Parquet file into Dask DataFrame
    print("Loading inference results from parquet...")
    infernece_df = dd.read_parquet("data/inference_results.parquet")

    #print(infernece_df.head())
    # Clear collection before upload
    print("Clearing old records from MongoDB...")
    mongo_client = MongoClient(mongo_uri)
    
    print(f"existing records {mongo_client[DATABASE][INFERENCE_COLLECTION].count_documents({})}")
    mongo_client[DATABASE][INFERENCE_COLLECTION].delete_many({})
    print(mongo_client[DATABASE][INFERENCE_COLLECTION].count_documents({}))
    print("Writing inference results to MongoDB...")
    write_dask_dataframe_to_mongodb(infernece_df, INFERENCE_COLLECTION, DATABASE, mongo_uri)

    # Close Dask client
    client.close()