import dask
from dask_ml.xgboost import XGBClassifier as xgb 
from dask.distributed import Client
import joblib 
import os 
import pandas as pd
import dask.dataframe as dd
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import json

def xgboost_optimize_hyperparameters(X_train_file, y_train_file, col_names, output_file, frac=0.2, max_evals=2):
    #read from parquet file
    X_train = dd.read_parquet(X_train_file)

    #load pickle y_train
    y_train = joblib.load(y_train_file)

    X_train.columns = col_names
    X_train['Target'] = y_train
    X_train = X_train.dropna()
    y_train = X_train.pop("Target")

    # Sample a subset of the data
    X_train_sample = X_train.sample(frac=frac, random_state=42).compute()
    y_train_sample = y_train.sample(frac=frac, random_state=42).compute()

    # Convert to pandas DataFrames
    X_train_sample_pd = pd.DataFrame(X_train_sample)
    y_train_sample_pd = pd.DataFrame(y_train_sample)

    def objective(params):
        params = {
            'n_estimators': int(params['n_estimators']), 
            'max_depth': int(params['max_depth']), 
            'learning_rate': params['learning_rate']
        }
        
        clf = xgb.XGBClassifier(**params)
        score = cross_val_score(clf, X_train_sample_pd, y_train_sample_pd, cv=5).mean()
        return {'loss': -score, 'status': STATUS_OK}

    space = {
        'n_estimators': hp.quniform('n_estimators', 50, 1000, 1),
        'max_depth': hp.quniform('max_depth', 1, 13, 1),
        'learning_rate': hp.loguniform('learning_rate', -5, 0)
    }

    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=max_evals)

    print("Hyperopt estimated optimum {}".format(best))

    # Save as json best parameters
    with open(output_file, 'w') as fp:
        json.dump(best, fp)

# You can call the function like this:
col_names = ['UDI', 'Product ID', 'Type', 'Air temperature', 'Process temperature', 'Rotational speed', 'Torque', 'Tool wear', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
optimize_hyperparameters("data/processed_data.parquet", "data/y_train.pkl", col_names, 'best_params.json')
