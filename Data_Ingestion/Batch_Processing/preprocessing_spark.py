from abc import abstractmethod
import dask.dataframe as dd
import numpy as np
import json 
from CONFIG_FILE import *
from Scaling import *
from Encoding import *
from Imputation import * 

from Saver import * 

import dotenv
dotenv.load_dotenv()


from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year, month, dayofmonth, hour, minute, second, dayofweek, dayofyear, weekofyear
from pyspark.ml.feature import StringIndexer, OneHotEncoder, Imputer, StandardScaler
from pyspark.ml import Pipeline
from pyspark.sql.types import IntegerType, FloatType

spark = SparkSession.builder.appName('data_processing').getOrCreate()

class SparkDataIngestion:
    
    def __init__(self):
        with open(CONFIG_FILE) as json_file:
            data = json.load(json_file)
            self.FILE_PATH = data["FILE_PATH"]
            self.TARGETCOL = data["TARGET_NAME"]
            self.DATECOL = data["DATECOL"]
            self.CATCOLS = data["CATCOLS"]
            self.NUMCOLS = data["NUMCOLS"]
            self.COLUNMS_TO_DROP = data["COLUNMS_TO_DROP"]
            self.PROCESSEDDATAPATH = data["PROCESSEDDATAPATH"]

            self.df = spark.read.csv(self.FILE_PATH, header=True, inferSchema=True)

    def _date_processing(self):
        for date_col in self.DATECOL:
            self.df = self.df.withColumn(date_col, col(date_col).cast('timestamp'))
            self.df = self.df.withColumn(date_col + "_year", year(col(date_col)))\
                             .withColumn(date_col + "_month", month(col(date_col)))\
                             .withColumn(date_col + "_day", dayofmonth(col(date_col)))\
                             .withColumn(date_col + "_hour", hour(col(date_col)))\
                             .withColumn(date_col + "_minute", minute(col(date_col)))\
                             .withColumn(date_col + "_second", second(col(date_col)))\
                             .withColumn(date_col + "_dayofweek", dayofweek(col(date_col)))\
                             .withColumn(date_col + "_dayofyear", dayofyear(col(date_col)))\
                             .withColumn(date_col + "_weekofyear", weekofyear(col(date_col)))
        return self.df

    def _split_data(self,test_size_ratio:float=0.2):
        self.train_df, self.test_df = self.df.randomSplit([1.0 - test_size_ratio, test_size_ratio], seed=123)
        return self.train_df, self.test_df

    def _scaling_numerical(self,type="standard"):
        stages = []
        for numeric_col in self.NUMCOLS:
            vectorAssembler = VectorAssembler(inputCols=[numeric_col], outputCol=numeric_col + "_Vect")
            scaler = StandardScaler(inputCol=numeric_col + "_Vect", outputCol=numeric_col + "_Scaled")
            stages += [vectorAssembler, scaler]
        pipeline = Pipeline(stages=stages)
        self.train_df = pipeline.fit(self.train_df).transform(self.train_df)
        self.test_df = pipeline.fit(self.test_df).transform(self.test_df)
        return self.train_df, self.test_df

    def _encoding_categorical(self, type="LabelEncoder"):
        indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(self.df) for column in self.CATCOLS]
        pipeline = Pipeline(stages=indexers)
        self.df = pipeline.fit(self.df).transform(self.df)
        return self.df

    def _handle_missing_values(self):
        imputer = Imputer(inputCols=self.NUMCOLS, outputCols=[f"{col}_imputed" for col in self.NUMCOLS])
        self.df = imputer.setStrategy("mean").fit(self.df).transform(self.df)
        return self.df

    def _drop_unnecessary_columns(self):
        for col in self.COLUNMS_TO_DROP:
            self.df = self.df.drop(col)
        return self.df

    def run_pipeline(self):
        self._date_processing()
        self._encoding_categorical()
        self._handle_missing_values()
        self._drop_unnecessary_columns()
        self._split_data()
        self._scaling_numerical()
        self.df.write.parquet(self.PROCESSEDDATAPATH)

if __name__ == "__main__":
    data_ingestion = SparkDataIngestion()
    