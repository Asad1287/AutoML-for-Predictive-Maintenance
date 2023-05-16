from sklearn.impute import SimpleImputer, KNNImputer

class Imputation:
    def __init__(self, df_train, df_test):
        self.df_train = df_train
        self.df_test = df_test

    def mean_imputation(self):
        mean_imputer = SimpleImputer(strategy='mean')
        df_train_imputed = self.df_train.copy()
        df_test_imputed = self.df_test.copy()
        df_train_imputed.iloc[:, :] = mean_imputer.fit_transform(df_train_imputed)
        df_test_imputed.iloc[:, :] = mean_imputer.transform(df_test_imputed)
        return df_train_imputed, df_test_imputed
    
    def mean_imputation_dask(self):
        mean_imputer = SimpleImputer(strategy='mean')
        df_train_imputed = self.df_train.map_partitions(mean_imputer.fit_transform)
        df_test_imputed = self.df_test.map_partitions(mean_imputer.transform)
        return df_train_imputed, df_test_imputed

    def mode_imputation_dask(self):
        mean_imputer = SimpleImputer(strategy='mode')
        df_train_imputed = self.df_train.map_partitions(mean_imputer.fit_transform)
        df_test_imputed = self.df_test.map_partitions(mean_imputer.transform)
        return df_train_imputed, df_test_imputed
    
    def median_imputation_dask(self):
        mean_imputer = SimpleImputer(strategy='median')
        df_train_imputed = self.df_train.map_partitions(mean_imputer.fit_transform)
        df_test_imputed = self.df_test.map_partitions(mean_imputer.transform)
        return df_train_imputed, df_test_imputed
    
    def median_imputation(self):
        median_imputer = SimpleImputer(strategy='median')
        df_train_imputed = self.df_train.copy()
        df_test_imputed = self.df_test.copy()
        df_train_imputed.iloc[:, :] = median_imputer.fit_transform(df_train_imputed)
        df_test_imputed.iloc[:, :] = median_imputer.transform(df_test_imputed)
        return df_train_imputed, df_test_imputed

    def mode_imputation(self):
        mode_imputer = SimpleImputer(strategy='most_frequent')
        df_train_imputed = self.df_train.copy()
        df_test_imputed = self.df_test.copy()
        df_train_imputed.iloc[:, :] = mode_imputer.fit_transform(df_train_imputed)
        df_test_imputed.iloc[:, :] = mode_imputer.transform(df_test_imputed)
        return df_train_imputed, df_test_imputed

    def knn_imputation(self, n_neighbors=3):
        knn_imputer = KNNImputer(n_neighbors=n_neighbors)
        df_train_imputed = self.df_train.copy()
        df_test_imputed = self.df_test.copy()
        df_train_imputed.iloc[:, :] = knn_imputer.fit_transform(df_train_imputed)
        df_test_imputed.iloc[:, :] = knn_imputer.transform(df_test_imputed)
        return df_train_imputed, df_test_imputed