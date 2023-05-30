from sklearn.preprocessing import MinMaxScaler, StandardScaler
from dask_ml.preprocessing import StandardScaler as DaskStandardScaler
from dask_ml.preprocessing import MinMaxScaler as DaskMinMaxScaler
import joblib
import pandas as pd 
class DataScaler:
    def __init__(self, df_train):
        self.df_train = df_train
        self.numerical_columns = df_train.select_dtypes(include=['float64', 'int64']).columns

    def fit(self, scaler_type='standard'):
        scaler = DaskStandardScaler() if scaler_type == 'standard' else DaskMinMaxScaler()
        # Convert to pandas DataFrame for fitting
        df_train_pd = self.df_train[self.numerical_columns].compute()
        scaler.fit(df_train_pd)
        # Save the scaler into a file using joblib
        joblib.dump(scaler, f'{scaler_type}_scaler.joblib')

    # Helper function for applying the scaler
    def _apply_scaler(self, x, scaler):
        return pd.DataFrame(scaler.transform(x), columns=x.columns, index=x.index)

    def transform(self, df, scaler_type='standard'):
        # Load the scaler from the file
        scaler = joblib.load(f'{scaler_type}_scaler.joblib')
        # Use map_partitions to apply the scaler to dask DataFrame
        meta = pd.DataFrame(columns=df.columns, dtype=float)  # specify meta here
        df_scaled = df.map_partitions(self._apply_scaler, scaler, meta=meta)
        return df_scaled

    def fit_transform(self, scaler_type='standard'):
        self.fit(scaler_type)
        return self.transform(self.df_train, scaler_type)
