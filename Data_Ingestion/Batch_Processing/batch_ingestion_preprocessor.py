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
import glob
from Saver import * 
from dask_ml.model_selection import train_test_split
import logging
import dotenv
dotenv.load_dotenv()

class DataIngestionProcessing:

    
    @abstractmethod
    def set_index(df: dd.DataFrame) -> dd.DataFrame:
        """
        The function sets the index of the dataframe

        args:
            df: dask dataframe
        
        
        """
        df = df.reset_index(drop=True)
        return df

    @abstractmethod
    def optimize_memory_df(df: dd.DataFrame) -> dd.DataFrame:
        """
        The function optimizes the memory of the dataframe by changing the data types of the columns

        args:
            df: dask dataframe
        """
        float_cols = df.select_dtypes(include=['float64']).columns.tolist()
        for float_col in float_cols:
            df[float_col] = df[float_col].astype(np.float32)
        int_cols = df.select_dtypes(include=['int64']).columns.tolist()
        for int_col in int_cols:
            df[int_col] = df[int_col].astype(np.int32)
        object_cols = df.select_dtypes(include=['object']).columns.tolist()
        for object_col in object_cols:
            df[object_col] = df[object_col].astype('category')
        return df
    
    def date_processing(df: dd.DataFrame,date_columns:List[str]) -> dd.DataFrame:
        """
        
        """
        if date_columns:
            for date_col in date_columns:
                df[date_col] = dd.to_datetime(df[date_col])
                df[date_col + "_year"] = df[date_col].dt.year
                df[date_col + "_month"] = df[date_col].dt.month
                df[date_col + "_day"] = df[date_col].dt.day
                df[date_col + "_hour"] = df[date_col].dt.hour
                df[date_col + "_minute"] = df[date_col].dt.minute
                df[date_col + "_second"] = df[date_col].dt.second
                df[date_col + "_dayofweek"] = df[date_col].dt.dayofweek
                df[date_col + "_week"] = df[date_col].dt.week
                df[date_col + "_weekofyear"] = df[date_col].dt.weekofyear
                df[date_col + "_dayofyear"] = df[date_col].dt.dayofyear
            return df
    @abstractmethod
    def split_data(df:dd.DataFrame,target_column:str,test_size_ratio:float=0.2) -> dd.DataFrame:
        """
        The function splits the data into train and test set

        args:
            df: dask dataframe
            target_column: target column name
            test_size_ratio: test size ratio
        """
        X = df.drop(target_column,axis=1)
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_ratio, random_state=123)
        
        return X_train, X_test, y_train, y_test
    
    @abstractmethod
    def scaling_numerical(X_train:pd.DataFrame,X_test:pd.DataFrame,type:str="standard") -> pd.DataFrame:
        """
        The function scales the numerical columns of the dataframe

        args:
            X_train: train dataframe
            X_test: test dataframe
            type: type of scaling
        """

        scalar = DataScaler(X_train,X_test)
        
        if type not in ["standard","minmax"]:
            raise ValueError("Scaling type must be standard or minmax")
        if type == "standard":
            X_train, X_test = scalar.standard_scaling_numpy()
        elif type == "MaxMin":
            X_train, X_test = scalar.min_max_scaling_numpy()
        return X_train, X_test
    
    @abstractmethod
    def encoding_categorical(X_train:pd.DataFrame,X_test:pd.DataFrame,CATCOLS:List[str],DATECOL:List[str],type:str="LabelEncoder") -> pd.DataFrame:
        """
        The function encodes the categorical columns of the dataframe

        args:
            X_train: train dataframe
            X_test: test dataframe
            CATCOLS: list of categorical columns
            DATECOL: list of date columns
            type: type of encoding
        """

        encoder = Encoding(X_train,X_test,CATCOLS,DATECOL)
        if type == "LabelEncoder":
            X_train, X_test = encoder.label_encoding_dask()
        elif type == "OneHotEncoder":
            pass
        return X_train, X_test
    
    @abstractmethod
    def imputation(X_train:pd.DataFrame,X_test:pd.DataFrame,NUMCOLS:List[str],CATCOLS:List[str],type:str="mean") -> pd.DataFrame:
        """
        The function imputes the missing values of the dataframe

        args:
            X_train: train dataframe
            X_test: test dataframe
            NUMCOLS: list of numerical columns
            CATCOLS: list of categorical columns
            type: type of imputation
        """
        imputer = Imputation(X_train,X_test,NUMCOLS,CATCOLS)
        if type == "mean":
            X_train, X_test = imputer.mean_imputation_dask()
        elif type == "median":
            X_train, X_test = imputer.median_imputation_dask()
        elif type == "mode":
            X_train, X_test = imputer.mode_imputation_dask()
        return X_train, X_test

   

class DataIngestion:
    def __init__(self, config_file:str, source_type:str, save_source_type:str, connection_string:str, query:str=None, csv_file_path:str=None,csv_folder_path:str=None):
        #CONFIG_FILE,"csv","pickle",None,None)
        self.source_type = source_type
        self.save_source = save_source_type
        self.connection_string = connection_string
        self.query = query
        self.csv_file_path = csv_file_path
        self.config_file = config_file
        self.delimiter = ","
        self.csv_folder_path = csv_folder_path
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
        


    
    def load_data(self) -> dd.DataFrame:

        if self.source_type == 'csv_folder':
            dfs = []
            for file in glob.glob(self.csv_folder_path + f'/*.{self.source_type}'):
                df = dd.read_csv(file, delimiter=self.delimiter)
                dfs.append(df)
            self.df =  dd.concat(dfs, axis=0)
            assert all(col in self.df.columns for col in self.DATECOL), "DATECOL not in dataframe"
            assert all(col in self.df.columns for col in self.CATCOLS), "CATCOLS not in dataframe"
            if isinstance(self.COLUNMS_TO_DROP, list) and self.COLUNMS_TO_DROP:
                self.df = self.df.drop(self.COLUNMS_TO_DROP, axis=1)
                return self.df
        elif self.source_type== 'csv':
            self.df = dd.read_csv(self.FILE_PATH)
            assert all(col in self.df.columns for col in self.DATECOL), "DATECOL not in dataframe"
            assert all(col in self.df.columns for col in self.CATCOLS), "CATCOLS not in dataframe"
            if isinstance(self.COLUNMS_TO_DROP, list) and self.COLUNMS_TO_DROP:
                self.df = self.df.drop(self.COLUNMS_TO_DROP, axis=1)
        elif self.source_type == 'postgres':
            engine = create_engine(self.connection_string)
            self.df = dd.from_pandas(pd.read_sql(self.query, engine), npartitions=2)
            assert all(col in self.df.columns for col in self.DATECOL), "DATECOL not in dataframe"
            assert all(col in self.df.columns for col in self.CATCOLS), "CATCOLS not in dataframe"
            if isinstance(self.COLUNMS_TO_DROP, list) and self.COLUNMS_TO_DROP:
                self.df = self.df.drop(self.COLUNMS_TO_DROP, axis=1)

        elif self.source_type == 'aws_redshift':
            engine = create_engine(self.connection_string)
            self.df =  dd.from_pandas(pd.read_sql(self.query, engine), npartitions=2)
            assert all(col in self.df.columns for col in self.DATECOL), "DATECOL not in dataframe"
            assert all(col in self.df.columns for col in self.CATCOLS), "CATCOLS not in dataframe"
            if isinstance(self.COLUNMS_TO_DROP, list) and self.COLUNMS_TO_DROP:
                self.df = self.df.drop(self.COLUNMS_TO_DROP, axis=1)
        elif self.source_type == 'sqlite':
            engine = create_engine(self.connection_string)
            self.df =  dd.from_pandas(pd.read_sql(self.query, engine), npartitions=2)
            assert all(col in self.df.columns for col in self.DATECOL), "DATECOL not in dataframe"
            assert all(col in self.df.columns for col in self.CATCOLS), "CATCOLS not in dataframe"
            if isinstance(self.COLUNMS_TO_DROP, list) and self.COLUNMS_TO_DROP:
                self.df = self.df.drop(self.COLUNMS_TO_DROP, axis=1)
        else:
            raise Exception("Invalid source type")
        
    def get_n_head(self, n:int=5) -> dd.DataFrame:
        return self.df.head(n)
    
    def save_data(self, df: dd.DataFrame,fname:str="df") -> None:
        
        if self.source_type == 'csv':
            save_file_path = os.path.join(self.PROCESSEDDATAPATH, f"{fname}.csv")
            df.to_csv(save_file_path, single_file = True)
        elif self.source_type == 'postgres':
            engine = create_engine(self.connection_string)
            df.to_sql(self.table_name, engine, if_exists='replace', index=False)
        elif self.source_type == 'aws_redshift':
            engine = create_engine(self.connection_string)
            df.to_sql(self.table_name, engine, if_exists='replace', index=False)
        elif self.source_type == 'sqlite':
            engine = create_engine(self.connection_string)
            df.to_sql(self.table_name, engine, if_exists='replace', index=False)
        elif self.source_type == 'pickle':
            df.to_pickle(self.csv_file_path)
        
        
        else:
            raise ValueError("Invalid source type provided.")
    

    

