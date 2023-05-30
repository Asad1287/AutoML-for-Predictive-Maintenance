import dask
from dask_ml.xgboost import XGBClassifier as xgb 
from dask.distributed import Client
import joblib 
import os 
import pandas as pd
import dask.dataframe as dd
import pickle
import warnings
import json
import dask_xgboost as dxgb

warnings.filterwarnings('ignore')

def xgboost_model(X_train_file, y_train_file, params_file, model_output_file, col_names):
    # Read from parquet file
    X_train = dd.read_parquet(X_train_file)
    X_train.columns = col_names

    # Load pickle y_train
    y_train = joblib.load(y_train_file)

    # Load from best_params.json
    with open(params_file) as json_file:
        best = json.load(json_file)

    # Create a LocalCluster instance
    client = Client()

    # Train XGBoost with best parameters from Hyperopt
    params = {'n_estimators': int(best['n_estimators']), 'max_depth': int(best['max_depth']), 'learning_rate': best['learning_rate']}
    model = dxgb.train(client, params, X_train, y_train)

    # Save the model to disk
    with open(model_output_file, "wb") as f:
        pickle.dump(model, f)

    # Load model from pickle
    with open(model_output_file, "rb") as f:
        print("Loading model from pickle...")
        model = pickle.load(f)
        print("Model loaded from pickle.")
    print(model)

    client.close()

# You can call the function like this:
col_names =['UDI', 'Product ID', 'Type', 'Air temperature', 'Process temperature', 'Rotational speed', 'Torque', 'Tool wear', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
xgboost_model("data/processed_data.parquet", "data/y_train.pkl", 'best_params.json', "model_xgb_best1.pkl", col_names)
