
from dask_ml.model_selection import train_test_split
import pandas as pd
from sklearn.utils.multiclass import type_of_target

from dask_ml.impute import SimpleImputer

import pandas as pd
import numpy as np
from typing import List

import pandas as pd
from dask import dataframe as dd

from src.Core_Classes.Balancer import *
from src.Core_Classes.DataScalar import *
from src.Core_Classes.Encoding import *
from src.Core_Classes.Imputation import *
from src.utils import *
from src.Core_Classes.Imbalance import * 
from src.MongoHandler import *
from src.Core_Classes.CustomScalar import *
import dill
import os
import joblib
import dotenv
from src.CONSTANTS import *
dotenv.load_dotenv()
import sys 
import os
PASSWORD = os.getenv("PASSWORD")
#database = sys.argv[1]
#collection = sys.argv[2]
#print(database)
#print(collection)


client = MongoDBHandler(MONGO_URI,f"{db_name}",f"{collection_name_training}")



import dask
import pandas as pd
from dask_ml.preprocessing import StandardScaler

import dask.dataframe as dd

import dask.dataframe as dd
import numpy as np


def process_data(client:MongoDBHandler,Train:bool=True) -> None:
    

    df = client.read_to_dask()
    df = df.drop("_id",axis=1)
    

    col_names = df.columns

    utils = Utils(DATECOL, TARGETCOL)

    list_of_func = [utils.set_index, utils.optimize_memory_df, utils.date_processing]
                    

    for func in list_of_func:
        df = func(df)

    if Train: 
        X_train, X_validation, y_train, y_validation = utils.split_data(df,test_size_ratio=0.2)
        scalar = CustomDaskMinMaxScaler()
        X_train = scalar.fit_transform(X_train)

        X_validation = scalar.transform(X_validation)

        enc = Encoding(X_train,CATCOLS)
        X_train = enc.label_encoding_fit()
        X_validation = enc.label_encoding_transform(X_validation)

        imputer = Imputation(X_train)
        X_train = imputer.fit_transform()
        X_validation = imputer.transform(X_validation)
        #save to parquet
        X_train.to_parquet(os.path.join(FEATURE_STORE, "processed_data.parquet"),engine="pyarrow")
        X_validation.to_parquet(os.path.join(FEATURE_STORE, "processed_data_validation.parquet"),engine="pyarrow")
        #pickle y_train
        joblib.dump(y_train,os.path.join(FEATURE_STORE,"y_train.pkl"))
        joblib.dump(y_validation,os.path.join(FEATURE_STORE,"y_validation.pkl"))

        joblib.dump(scalar,os.path.join(FEATURE_STORE,"scalar.pkl"))
        enc_object = dill.dumps(enc)
        imputer_object = dill.dumps(imputer)

        # Save the serialized object to a file
        with open(os.path.join(FEATURE_STORE,"enc.pkl"), "wb") as f:
            f.write(enc_object)
        with open(os.path.join(FEATURE_STORE,"imputer.pkl"), "wb") as f:
            f.write(imputer_object)

    else:
        # load the pickle objects encoder, imputer ,scalar
        try:
            scalar = joblib.load(os.path.join(FEATURE_STORE,"scalar.pkl"))
        except:
            raise FileNotFoundError("No encoder object found")
        try:
            enc = dill.load(open(os.path.join(FEATURE_STORE,"enc.pkl")), "rb")
        except:
            raise FileNotFoundError("No encoder object found")
        try:
            imputer = dill.load(open(os.path.join(FEATURE_STORE,"imputer.pkl"), "rb"))
        except:
            raise FileNotFoundError("No encoder object found")
        try:
            df = scalar.transform(df)
            df = enc.label_encoding_transform(df)
            df = imputer.transform(df)

            df.to_parquet(os.path.join(FEATURE_STORE,"data/processed_data_test.parquet"),engine="pyarrow")
        #print error if any
        except  Exception as e:
            print(e)
            raise Exception("Error in processing data")
           
        return df 


   


   
if __name__ == "__main__":
    process_data(client)
    #client = MongoDBHandler("mongodb://localhost:27017/","predictive_maintenance","predictive_maintenance")
    #client.insert_many("predictive_maintenance",df)
    #client.read_
#client.search_by_product_id("predictive_maintenance","L47181")

