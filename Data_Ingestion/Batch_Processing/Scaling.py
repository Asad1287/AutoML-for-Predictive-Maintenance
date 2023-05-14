from sklearn.preprocessing import MinMaxScaler, StandardScaler
from dask_ml.preprocessing import StandardScaler as DaskStandardScaler
from dask_ml.preprocessing import MinMaxScaler as DaskMinMaxScaler
class DataScaler:
    def __init__(self, df_train, df_test):
        self.df_train = df_train
        self.df_test = df_test

    def min_max_scaling(self):
        min_max_scaler = MinMaxScaler()
        df_train_scaled = self.df_train.copy()
        df_test_scaled = self.df_test.copy()
        df_train_scaled.iloc[:, :] = min_max_scaler.fit_transform(df_train_scaled)
        df_test_scaled.iloc[:, :] = min_max_scaler.transform(df_test_scaled)
        return df_train_scaled, df_test_scaled

    def standard_scaling(self):
        standard_scaler = StandardScaler()
        df_train_scaled = self.df_train.copy()
        df_test_scaled = self.df_test.copy()
        df_train_scaled.iloc[:, :] = standard_scaler.fit_transform(df_train_scaled)
        df_test_scaled.iloc[:, :] = standard_scaler.transform(df_test_scaled)
        return df_train_scaled, df_test_scaled
    
    def standard_scaling_dask(self):
        standard_scaler = DaskStandardScaler()
        df_train_scaled = self.df_train.map_partitions(standard_scaler.fit_transform)
        df_test_scaled = self.df_test.map_partitions(standard_scaler.transform)
        return df_train_scaled, df_test_scaled
    def min_max_scaling_dask(self):
        min_max_scaler = DaskMinMaxScaler()
        df_train_scaled = self.df_train.map_partitions(min_max_scaler.fit_transform)
        df_test_scaled = self.df_test.map_partitions(min_max_scaler.transform)
        return df_train_scaled, df_test_scaled
    
    def standard_scaling_numpy(self):
        standard_scaler = StandardScaler()
        df_train_scaled = standard_scaler.fit_transform(self.df_train)
        df_test_scaled = standard_scaler.transform(self.df_test)
        return df_train_scaled, df_test_scaled
    
    def min_max_scaling_numpy(self):
        min_max_scaler = MinMaxScaler()
        df_train_scaled = min_max_scaler.fit_transform(self.df_train)
        df_test_scaled = min_max_scaler.transform(self.df_test)
        return df_train_scaled, df_test_scaled