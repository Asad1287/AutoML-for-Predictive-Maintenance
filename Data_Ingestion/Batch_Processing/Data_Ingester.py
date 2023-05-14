import pandas as pd
import dask.dataframe as dd
from pyspark.sql import SparkSession

class DataIngestion:
    def __init__(self, engine="pandas"):
        self.engine = engine
        if engine == "pyspark":
            self.spark = SparkSession.builder \
                .master("local") \
                .appName("DataIngestion") \
                .getOrCreate()

    def read_file(self, file_path, file_format, **kwargs):
        if self.engine == "pandas":
            if file_format == "csv":
                return pd.read_csv(file_path, **kwargs)
            elif file_format == "txt":
                return pd.read_csv(file_path, **kwargs)
            elif file_format == "feather":
                return pd.read_feather(file_path, **kwargs)
            elif file_format == "json":
                return pd.read_json(file_path, **kwargs)
            elif file_format == "parquet":
                return pd.read_parquet(file_path, **kwargs)
            else:
                raise ValueError("Unsupported file format")
        elif self.engine == "dask":
            if file_format == "csv":
                return dd.read_csv(file_path, **kwargs)
            elif file_format == "txt":
                return dd.read_csv(file_path, **kwargs)
            elif file_format == "json":
                return dd.read_json(file_path, **kwargs)
            elif file_format == "parquet":
                return dd.read_parquet(file_path, **kwargs)
            else:
                raise ValueError("Unsupported file format")
        elif self.engine == "pyspark":
            if file_format == "csv":
                return self.spark.read.csv(file_path, **kwargs)
            elif file_format == "txt":
                return self.spark.read.text(file_path, **kwargs)
            elif file_format == "json":
                return self.spark.read.json(file_path, **kwargs)
            elif file_format == "parquet":
                return self.spark.read.parquet(file_path, **kwargs)
            else:
                raise ValueError("Unsupported file format")

    def read_sql_table(self, table_name, connection, query=None, **kwargs):
        if query is None:
            query = f"SELECT * FROM {table_name}"
        if self.engine == "pandas":
            return pd.read_sql_query(query, connection, **kwargs)
        elif self.engine == "dask":
            return dd.read_sql_table(table_name, connection, **kwargs)
        elif self.engine == "pyspark":
            return self.spark.read.jdbc(connection, table_name, **kwargs)


# For Pandas
#ingestion = DataIngestion(engine="pandas")
#df = ingestion.read_file("data.csv", "csv")

# For Dask
#ingestion = DataIngestion(engine="dask")
#df = ingestion.read_file("data.csv", "csv")

# For PySpark
#ingestion = DataIngestion(engine="pyspark")
#df = ingestion.read_file("data.csv", "csv")
