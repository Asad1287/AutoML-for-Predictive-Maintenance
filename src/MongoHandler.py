import dask.dataframe as dd
import pandas as pd
from pymongo import MongoClient
class MongoDBHandler:
    def __init__(self, connection_string:str,db_name:str,collection_name:str):
        self.client = MongoClient(connection_string)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.df = None

    def read_to_dask(self,partition_size:int=2):
        # Read data from MongoDB
        data = list(self.collection.find())
        # Create a pandas dataframe
        
        # Convert pandas dataframe to dask dataframe
        self.df = dd.from_pandas(pd.DataFrame(data), npartitions=partition_size)
        return self.df
    
    def read_to_pandas(self):
        # Read data from MongoDB
        data = list(self.collection.find())
        # Create a pandas dataframe
        self.df = pd.DataFrame(data)
        return self.df
    
    def write_from_pandas(self,temp_df:pd.DataFrame,collection_name:str) -> None:
        # Convert dask dataframe to pandas dataframe
        
        # Convert the DataFrame to a list of dict records
        data = temp_df.to_dict("records")
        # Insert the data into the collection
        collection_temp  = self.db[collection_name]
        collection_temp.insert_many(data)
    
    def write_from_dask(self,temp_df:dd.DataFrame,collection_name:str) -> None:
        # Convert dask dataframe to pandas dataframe
        
        # Convert the DataFrame to a list of dict records
        data = temp_df.to_dict("records")
        # Insert the data into the collection
        collection_temp  = self.db[collection_name]
        collection_temp.insert_many(data)
    
    def get_df(self):
        return self.df
    def set_df(self,df:dd.DataFrame) -> None:
        self.df = df

    def upload_csv(self, collection_name, csv_path):
        """
        Uploads a csv file at path csv_path to mongo collection
        named collection_name
        """
        data = pd.read_csv(csv_path)
        self.db[collection_name].insert_many(data.to_dict('records'))

    def search_by_product_id(self, collection_name, product_id):
        """
        Search for documents in collection_name by product_id
        Returns a list of matching documents.
        """
        return [doc for doc in self.db[collection_name].find({"Product ID": product_id})]

    def retrieve_document(self, collection_name, document_id):
        """
        Retrieve a single document from collection_name by its id
        """
        return self.db[collection_name].find_one({"_id": document_id})

    def load_to_dask_dataframe(self, collection_name):
        """
        Load a mongo collection into a Dask DataFrame
        """
        cursor = self.db[collection_name].find()
        df = pd.DataFrame(list(cursor))
        dask_df = dd.from_pandas(df, npartitions=3)
        return dask_df
    
    def clear_collection(self, collection_name):
        """
        Delete all documents from a collection
        """
        self.db[collection_name].delete_many({})
            