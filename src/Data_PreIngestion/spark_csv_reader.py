import os
from pyspark.sql import SparkSession

def read_all_csv_and_save_as_parquet(input_dir, output_path):
    # Create a SparkSession
    spark = SparkSession.builder.getOrCreate()

    # Get a list of all csv files in the directory
    input_files = [os.path.join(input_dir, filename) for filename in os.listdir(input_dir) if filename.endswith(".csv")]

    # Read data from multiple CSV files
    df = spark.read.format('csv').option('header','true').load(input_files)

    # Save DataFrame as Parquet
    df.write.parquet(output_path)

# Usage:
input_dir = "/path/to/your/csv/files/directory"

read_all_csv_and_save_as_parquet(input_dir, "/path/to/save/parquet/files")
