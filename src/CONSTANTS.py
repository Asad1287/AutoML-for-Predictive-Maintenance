import os 
import dotenv
dotenv.load_dotenv()
import sys 
import os
import os 
import dotenv
import argparse
from column_types import *
dotenv.load_dotenv()

# create parser
parser = argparse.ArgumentParser()

# add arguments to the parser
parser.add_argument('-d', '--data_path', type=str, help='The data path.')
parser.add_argument('-p', '--password', type=str, help='Password for MongoDB.')
parser.add_argument('-f', '--feature_store', type=str, help='The feature store path.')
parser.add_argument('-db', '--database_name', type=str, help='The database name.')
parser.add_argument('-ct', '--collection_name_training', type=str, help='The training collection name.')
parser.add_argument('-mu', '--mongo_uri', type=str, help='The MongoDB URI.')
#add inference collection name
parser.add_argument('-ci', '--collection_name_inference', type=str, help='The inference collection name.')

args = parser.parse_args()

PASSWORD = args.password or os.getenv("PASSWORD")
FEATURE_STORE = args.feature_store or "/mnt/d/Portfolio/Auto_ML_Pdm/AutoML/src/FeatureStore"
db_name = args.database_name or "predictive_maintenance"
collection_name_training = args.collection_name_training or "predictive_maintenance"
MONGO_URI = args.mongo_uri or f"mongodb+srv://root12345:{PASSWORD}@cluster1.b03tix4.mongodb.net/"


INFERENCE_COLLECTION = args.collection_name_inference or "inference_results"