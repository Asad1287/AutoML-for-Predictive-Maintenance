from abc import abstractmethod
import dask.dataframe as dd
import numpy as np
import json 
from CONFIG_FILE import *
from Scaling import *
from Encoding import *
from Imputation import * 
import joblib
import os 
from Saver import * 
from dask_ml.model_selection import train_test_split
import logging
import dotenv
dotenv.load_dotenv()

class DataIngestion:
    def __init__(self, engine="dask"):
        with open(CONFIG_FILE) as json_file:
            data = json.load(json_file)
            self.FILE_PATH = data["FILE_PATH"]
            self.TARGETCOL = data["TARGET_NAME"]
            assert self.TARGETCOL, "TARGETCOL is empty"
            assert isinstance(self.TARGETCOL, str), "TARGETCOL is not a string"
            self.DATECOL = data["DATECOL"]
            assert isinstance(self.DATECOL, list), "DATECOL is not a list"
            self.CATCOLS = data["CATCOLS"]
            assert isinstance(self.CATCOLS, list), "CATCOLS is not a list"
            self.NUMCOLS = data["NUMCOLS"]
            assert isinstance(self.NUMCOLS, list), "NUMCOLS is not a list"
            self.COLUNMS_TO_DROP = data["COLUNMS_TO_DROP"]
            assert isinstance(self.COLUNMS_TO_DROP, list), "COLUNMS_TO_DROP is not a list"
            self.PROCESSEDDATAPATH = data["PROCESSEDDATAPATH"]
            assert isinstance(self.PROCESSEDDATAPATH, str), "PROCESSEDDATAPATH is not a string"
        
        self.df = dd.read_csv(self.FILE_PATH)
        assert all(col in self.df.columns for col in self.DATECOL), "DATECOL not in dataframe"
        assert all(col in self.df.columns for col in self.CATCOLS), "CATCOLS not in dataframe"
        if isinstance(self.COLUNMS_TO_DROP, list) and self.COLUNMS_TO_DROP:
            self.df = self.df.drop(self.COLUNMS_TO_DROP, axis=1)
    
    #set index for df , add an index column
    def _set_index(self):
        self.df = self.df.reset_index(drop=True)
        return self.df

    def _optimize_memory_df(self):
        float_cols = self.df.select_dtypes(include=['float64']).columns.tolist()
        for float_col in float_cols:
            self.df[float_col] = self.df[float_col].astype(np.float32)
        int_cols = self.df.select_dtypes(include=['int64']).columns.tolist()
        for int_col in int_cols:
            self.df[int_col] = self.df[int_col].astype(np.int32)
        object_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        for object_col in object_cols:
            self.df[object_col] = self.df[object_col].astype('category')
        return self.df
    
    def _date_processing(self):
        if self.DATECOL:
            for date_col in self.DATECOL:
                self.df[date_col] = dd.to_datetime(self.df[date_col])
                self.df[date_col + "_year"] = self.df[date_col].dt.year
                self.df[date_col + "_month"] = self.df[date_col].dt.month
                self.df[date_col + "_day"] = self.df[date_col].dt.day
                self.df[date_col + "_hour"] = self.df[date_col].dt.hour
                self.df[date_col + "_minute"] = self.df[date_col].dt.minute
                self.df[date_col + "_second"] = self.df[date_col].dt.second
                self.df[date_col + "_dayofweek"] = self.df[date_col].dt.dayofweek
                self.df[date_col + "_week"] = self.df[date_col].dt.week
                self.df[date_col + "_weekofyear"] = self.df[date_col].dt.weekofyear
                self.df[date_col + "_dayofyear"] = self.df[date_col].dt.dayofyear
            return self.df
    
    def _split_data(self,test_size_ratio:float=0.2):
        X = self.df.drop(self.TARGETCOL, axis=1)
        y = self.df[self.TARGETCOL]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size_ratio, random_state=123)
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def _scaling_numerical(self,type="standard"):
        scalar = DataScaler(self.X_train,self.X_test)
        
        if type not in ["standard","minmax"]:
            raise ValueError("Scaling type must be standard or minmax")
        if type == "standard":
            self.X_train, self.X_test = scalar.standard_scaling_numpy()
        elif type == "MaxMin":
            self.X_train, self.X_test = scalar.min_max_scaling_numpy()
        return self.X_train, self.X_test

    def _encoding_categorical(self, type="LabelEncoder"):
        encoder = Encoding(self.X_train, self.X_test, self.TARGETCOL, self.CATCOLS+ self.DATECOL)
        if type == "LabelEncoder":
            self.X_train, self.X_test = encoder.label_encoding_dask()
        elif type == "OneHotEncoder":
            pass
        return self.X_train, self.X_test

    def _data_imputation(self, type="mean"):
        imputer = Imputation(self.X_train, self.X_test)
        if type == "mean":
            self.X_train, self.X_test = imputer.mean_imputation_dask()
        elif type == "median":
            pass
        elif type == "mode":
            pass
        return self.X_train, self.X_test

    def _save_to_disk(self, path,type="pickle"):
        saver_obj = Saver(self.X_train, self.X_test, self.y_train, self.y_test)
        if type == "pickle":
            saver_obj.to_pickle_local(path)
            logging.info("Dataframe saved to disk in pickle format")
        elif type == "csv":
            saver_obj.save_csv(path)
            logging.info("Dataframe saved to disk in csv format")
        return True

    def process_df(self):
        self._set_index()
        self._optimize_memory_df()
        self._date_processing()
        self._split_data()
        self._encoding_categorical()
        self._data_imputation()
        self._scaling_numerical()
        
        return self.X_train, self.X_test, self.y_train, self.y_test

DataIngestion = DataIngestion()
X_train, X_test, y_train, y_test= DataIngestion.process_df()
#convert x_train to pandas dataframe
print(X_train.shape)
print(type(X_train))




