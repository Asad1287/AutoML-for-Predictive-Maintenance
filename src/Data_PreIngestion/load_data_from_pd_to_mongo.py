from pymongo import MongoClient
import pandas as pd
import os
import dotenv

def load_dataframe_to_mongodb(df:pd.DataFrame, collection_name:str, URI:str, database_name:str='predictive_maintenance', chunk_size:int=5000) -> None:
    # Load environment variables
    dotenv.load_dotenv()
    PASSWORD = os.getenv("PASSWORD")

    # Create a connection to MongoDB
    client = MongoClient(URI)

    # Access the database
    db = client[database_name]

    # Access the collection
    collection = db[collection_name]

    # Convert the DataFrame to a list of dict records and insert to collection
    data = df.to_dict("records")
    
    # Chunking data for efficient insertion
    if len(data) > chunk_size:
        print("Inserting data in chunks.")
        chunks = [data[x:x+chunk_size] for x in range(0, len(data), chunk_size)]
        for chunk in chunks:
            collection.insert_many(chunk)
    else:
        collection.insert_many(data)
        
    print("Data loaded successfully.")
