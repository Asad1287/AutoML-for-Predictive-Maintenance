from data_ingestion import *
import pickle
from dask.distributed import Client
import dask_xgboost as dxgb
from MongoHandler import *
from src.CONSTANTS import *


def prediction_pipeline(test_data_path, model_path, col_names):
    client = Client()

    # Load test data
    test_data = dd.read_parquet(test_data_path)
    test_data.columns = col_names

    test_data = test_data.reset_index(drop=True)

    # Load model from pickle
    with open(model_path, "rb") as f:
        print("Loading model from pickle...")
        model = pickle.load(f)
        print("Model loaded from pickle.")
    
    # Run prediction on test data
    y_pred = dxgb.predict(client, model, test_data).compute()
    
    y_pred = [0 if x < 0.5 else 1 for x in y_pred]
    y_pred_series = pd.Series(y_pred)
    
    test_data = test_data.compute()

    test_data['predictions'] = y_pred_series
    
    try:
        print("Uploading to mongo...")
        client_mongo = MongoDBHandler(f"mongodb+srv://root12345:{PASSWORD}@cluster1.b03tix4.mongodb.net/",f"{DATABASE}",f"{INFERENCE_COLLECTION}")
        client_mongo.clear_collection(INFERENCE_COLLECTION)
        client_mongo.write_from_pandas(test_data, INFERENCE_COLLECTION)
        print("Uploaded to mongo.")
    except Exception as e:
        print(f"Error in uploading to mongo: {e}")
    
    client.close()

if __name__ == '__main__':
    col_names =['UDI', 'Product ID', 'Type', 'Air temperature',
           'Process temperature', 'Rotational speed', 'Torque',
           'Tool wear', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']

    test_data_path = "data/processed_data_test.parquet"
    model_path = "model_xgb_best1.pkl"

    prediction_pipeline(test_data_path, model_path, col_names)
