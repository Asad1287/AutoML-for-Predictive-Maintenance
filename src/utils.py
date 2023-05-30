
import numpy as np
import pandas as pd
from dask_ml.model_selection import train_test_split
class Utils:
    def __init__(self, DATECOL, TARGETCOL):
        self.DATECOL =  DATECOL
        self.TARGETCOL = TARGETCOL

    def set_index(self,df):
            df = df.reset_index(drop=True)
            return df

    def optimize_memory_df(self,df):
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
        
    def date_processing(self,df):
            if self.DATECOL:
                for date_col in self.DATECOL:
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
            else:
                return df 
            
    def split_data(self,df,test_size_ratio:float=0.2):
            X = df.drop(self.TARGETCOL, axis=1)
            y = df[self.TARGETCOL]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_ratio, random_state=123)
            
            return X_train, X_test, y_train, y_test


#create  a function to write a list to a csv file 
def write_list_to_csv(list_to_write, csv_file_name):
    with open(csv_file_name, 'w') as f:
        for item in list_to_write:
            f.write("%s\n" % item)
import csv
#create a function to read a csv file into a list
def read_csv_to_list(csv_file_name):
    with open(csv_file_name, 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)
        #flatten your_list
        your_list = [item for sublist in your_list for item in sublist]
    return your_list

col_names = ["A","B","C"]
write_list_to_csv(col_names,"col_names.csv")
col_names = read_csv_to_list("col_names.csv")
print(col_names)


def optimize_memory_df(df):
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


