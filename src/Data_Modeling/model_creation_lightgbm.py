import dask.dataframe as dd
from lightgbm import LGBMClassifier
import joblib
import json
import pandas as pd
from sklearn.metrics import accuracy_score

def lightgbm_model(X_train_file, y_train_file, X_validation_file, y_validation_file, params_file, col_names, model_output_file):
    # Read from parquet file
    X_train = dd.read_parquet(X_train_file)
    X_validation = dd.read_parquet(X_validation_file)

    # Load pickle y_train
    y_train = joblib.load(y_train_file)
    y_validation = joblib.load(y_validation_file)

    X_train.columns = col_names

    # Load from best_params.json
    with open(params_file) as json_file:
        best = json.load(json_file)

    # Convert Dask DataFrame to Pandas DataFrame
    X_train_pd = X_train.compute()
    y_train_pd = pd.Series(y_train)
    X_validation_pd = X_validation.compute()
    y_validation_pd = pd.Series(y_validation)

    # Train LightGBM with best parameters from Hyperopt
    params = {'n_estimators': int(best['n_estimators']), 'max_depth': int(best['max_depth']), 'learning_rate': best['learning_rate']}
    model = LGBMClassifier(**params)

    model.fit(X_train_pd, y_train_pd)
    #predict on the validation set
    y_pred = model.predict(X_validation_pd)
    #evaluate the accuracy
    accuracy = accuracy_score(y_validation_pd, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    # Save the model to disk
    joblib.dump(model, model_output_file)

# You can call the function like this:
col_names =['UDI', 'Product ID', 'Type', 'Air temperature', 'Process temperature', 'Rotational speed', 'Torque', 'Tool wear', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
lightgbm_model("data/processed_data.parquet", "data/y_train.pkl", "data/processed_data_validation.parquet", "data/y_validation.pkl", 'best_params.json', col_names, 'model_lgbm_best1.pkl')
