from abc import abstractmethod
import pandas as pd
import os 
import pyarrow.parquet as pq
import boto3
import sqlite3
import psycopg2
from pymongo import MongoClient
from sqlalchemy import create_engine
import pyarrow as pa
import os
import pandas as pd
import boto3
import pickle
import joblib
from math import ceil


class Saver:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def to_csv(self, path):
        self.X_train.to_csv(path + 'X_train.csv')
        self.y_train.to_csv(path + 'y_train.csv')
        self.X_test.to_csv(path + 'X_test.csv')
        self.y_test.to_csv(path + 'y_test.csv')
    

    
    def to_pickle_local(self, path):
        joblib.dump(self.X_train, os.path.join(path, 'X_train.pkl'))
        joblib.dump(self.y_train, os.path.join(path, 'y_train.pkl'))
        joblib.dump(self.X_test, os.path.join(path, 'X_test.pkl'))
        joblib.dump(self.y_test, os.path.join(path, 'y_test.pkl'))


   

    def to_parquet(self, path):
        pq.write_table(pa.Table.from_pandas(self.X_train), path + 'X_train.parquet')
        pq.write_table(pa.Table.from_pandas(self.y_train), path + 'y_train.parquet')
        pq.write_table(pa.Table.from_pandas(self.X_test), path + 'X_test.parquet')
        pq.write_table(pa.Table.from_pandas(self.y_test), path + 'y_test.parquet')

    def to_sqlite(self, db_name):
        conn = sqlite3.connect(db_name)
        self.X_train.to_sql('X_train', conn)
        self.y_train.to_sql('y_train', conn)
        self.X_test.to_sql('X_test', conn)
        self.y_test.to_sql('y_test', conn)
        conn.close()
    
    @abstractmethod
    def save_to_s3(data, bucket, filename, aws_access_key_id, aws_secret_access_key, filetype='csv', split_size=100):
        s3 = boto3.resource('s3', 
                            aws_access_key_id=aws_access_key_id, 
                            aws_secret_access_key=aws_secret_access_key)
        
        if filetype == 'pickle':
            pickle_byte_obj = pickle.dumps(data) 
            s3.Object(bucket, filename + '.pkl').put(Body=pickle_byte_obj)
            
        elif filetype == 'csv':
            data.to_csv(filename + '.csv', index=False)
            filesize = os.path.getsize(filename + '.csv') / (1024 * 1024)  # size in MB

            if filesize > split_size:
                chunksize = ceil(len(data) / ceil(filesize / split_size))
                if not os.path.exists('temp'):
                    os.makedirs('temp')
                for i, chunk in enumerate(pd.read_csv(filename + '.csv', chunksize=chunksize)):
                    chunk.to_csv(f'temp/{filename}_{i}.csv', index=False)
                    s3.meta.client.upload_file(f'temp/{filename}_{i}.csv', bucket, f'{filename}_{i}.csv')
                    os.remove(f'temp/{filename}_{i}.csv')  # remove file after upload
                os.remove(filename + '.csv')  # remove original large file
                os.rmdir('temp')  # remove temporary directory
            else:
                s3.meta.client.upload_file(filename + '.csv', bucket, filename + '.csv')
                os.remove(filename + '.csv')  # remove file after upload
    
    @abstractmethod
    def save_to_mongo(df, db_name, collection_name, host='localhost', port=27017):
        client = MongoClient(host, port)
        db = client[db_name]
        collection = db[collection_name]

        # Convert DataFrame to list of dictionaries
        records = df.to_dict('records')

        # Insert records into collection
        collection.insert_many(records)
