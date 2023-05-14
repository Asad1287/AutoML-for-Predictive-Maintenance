from abc import abstractmethod
import pandas as pd 
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
from sklearn.model_selection import train_test_split
import logging


import dotenv
dotenv.load_dotenv()

class DataIngestion:
    
    """
    1. Reads data from csv format from csv
    2. Applies date processing on datetime column
    3. Applies encoding on catoegorical columns
    """
    def __init__(self, engine="pandas"):


       
        
        #load json file DF_INFO from path and extract info TARGETCOL, DATECOL, CATCOLS, NUMCOLS
               
        with open(CONFIG_FILE) as json_file:
            data = json.load(json_file)
            self.FILE_PATH = data["FILE_PATH"]
            self.TARGETCOL = data["TARGET_NAME"]
            #assert TARGETCOL is not empty and is a string 
            assert self.TARGETCOL, "TARGETCOL is empty"
            assert isinstance(self.TARGETCOL, str), "TARGETCOL is not a string"
            self.DATECOL = data["DATECOL"]
            #print type of DATECOL
            print(type(self.DATECOL))
            assert isinstance(self.DATECOL, list), "DATECOL is not a list"
            self.CATCOLS = data["CATCOLS"]
            assert isinstance(self.CATCOLS, list), "CATCOLS is not a list"
            self.NUMCOLS = data["NUMCOLS"]
            assert isinstance(self.NUMCOLS, list), "NUMCOLS is not a list"
            self.COLUNMS_TO_DROP = data["COLUNMS_TO_DROP"]
            assert isinstance(self.COLUNMS_TO_DROP, list), "COLUNMS_TO_DROP is not a list"
            self.PROCESSEDDATAPATH = data["PROCESSEDDATAPATH"]
            assert isinstance(self.PROCESSEDDATAPATH, str), "PROCESSEDDATAPATH is not a string"
        
        
            self.df = pd.read_csv(self.FILE_PATH)
            #check if list of self.DATECOL, self.CATCOLS,self.NUMCOLS, self.COLUNMS_TO_DROP columns name are in dataframe 
            assert all(col in self.df.columns for col in self.DATECOL), "DATECOL not in dataframe"
            assert all(col in self.df.columns for col in self.CATCOLS), "CATCOLS not in dataframe"

            #check if self.COLUMN_TO_DROP is a list and not empty then drop columns
            if isinstance(self.COLUNMS_TO_DROP, list) and self.COLUNMS_TO_DROP:
                self.df.drop(self.COLUNMS_TO_DROP, axis=1, inplace=True)
       
        
        
    def _optimize_memory_df(self):
        """
        Reduces memory usage of dataframe by converting float64 to float32 and int64 to int32
        convert object to category
        """
        #convert float64 to float32
        float_cols = self.df.select_dtypes(include=['float64']).columns.tolist()
        self.df[float_cols] = self.df[float_cols].astype(np.float32)
        #convert int64 to int32
        int_cols = self.df.select_dtypes(include=['int64']).columns.tolist()
        self.df[int_cols] = self.df[int_cols].astype(np.int32)
        #convert object to category
        object_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        self.df[object_cols] = self.df[object_cols].astype('category')
        return self.df
    
    def _date_processing(self):
        """
        Convert date column to datetime format and extract all date time related features
        such as year, month, day, hour, minute, second, day of week, day of year, week of year
        """
        if self.DATECOL:
            for date_col in self.DATECOL:
                self.df[date_col] = pd.to_datetime(self.df[date_col])
                self.df[date_col + "_year"] = self.df[date_col].dt.year
                self.df[date_col + "_month"] = self.df[date_col].dt.month
                self.df[date_col + "_day"] = self.df[date_col].dt.day
                self.df[date_col + "_hour"] = self.df[date_col].dt.hour
                self.df[date_col + "_minute"] = self.df[date_col].dt.minute
                self.df[date_col + "_second"] = self.df[date_col].dt.second
                self.df[date_col + "_dayofweek"] = self.df[date_col].dt.dayofweek
                self.df[date_col + "_dayofyear"] = self.df[date_col].dt.dayofyear
                self.df[date_col + "_weekofyear"] = self.df[date_col].dt.weekofyear
            return self.df
    
   

    def _split_data(self,test_size_ratio:float=0.2):
        """
        Split data into train and test
        """
        X = self.df.drop(self.TARGETCOL, axis=1)
        y = self.df[self.TARGETCOL]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size_ratio, random_state=123)
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def _scaling_numerical(self,type="standard"):
        """
        Apply scaling on numerical columns on self.X_train and self.X_test
        """
        scalar = DataScaler(self.X_train,self.X_test)
        if type not in ["standard","minmax"]:
            raise ValueError("Scaling type must be standard or minmax")
        if type == "standard":
            self.X_train, self.X_test = scalar.standard_scaling()
        elif type == "MaxMin":
            self.X_train, self.X_test = scalar.min_max_scaling()
        return self.X_train, self.X_test
    #write a abstract method for perfoming scaling on any numerical columns list in dataframe 
  
    
    def _encoding_categorical(self, type="LabelEncoder"):
        """
        Apply encoding on categorical columns on self.X_train and self.X_test
        """
        encoder = Encoding(self.X_train, self.X_test, self.TARGETCOL, self.CATCOLS+ self.DATECOL)
        if type == "LabelEncoder":
            self.X_train, self.X_test = encoder.label_encoding()
        elif type == "OneHotEncoder":
            pass
        return self.X_train, self.X_test

    def _data_imputation(self, type="mean"):
        """
        Apply imputation on self.X_train and self.X_test
        """
        imputer = Imputation(self.X_train, self.X_test)
        if type == "mean":
            self.X_train, self.X_test = imputer.mean_imputation()
        elif type == "median":
            pass
        elif type == "mode":
            pass
        return self.X_train, self.X_test

    def _save_to_disk(self, path,type="pickle"):
        """
        Save the processed data X_train, X_test,y_train,y_test to disk
        """
        saver_obj = Saver(self.X_train, self.X_test, self.y_train, self.y_test)
        if type == "pickle":
            saver_obj.to_pickle_local(path)
            logging.info("Dataframe saved to disk in pickle format")
            

           
            
        elif type == "csv":
            saver_obj.save_csv(path)
            logging.info("Dataframe saved to disk in csv format")
        return True

    def process_df(self):
        """
        Run the processing pipeline
        1. Optimize memory usage
        2. Date processing
        3. Split data
        4. Imputation
        5. Scaling
        6. Encoding
        7. Save to disk
        """
        self._optimize_memory_df()
        self._date_processing()
        self._split_data()
        self._encoding_categorical()
        self._data_imputation()
        self._scaling_numerical()
        self._save_to_disk(self.PROCESSEDDATAPATH)
        
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    


DataIngestion = DataIngestion()
X_train, X_test, y_train, y_test= DataIngestion.process_df()
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print(X_train.head())


    