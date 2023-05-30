from pyspark.ml.feature import VectorAssembler, VectorSlicer
from pyspark.sql import SparkSession

def process_data(input_path: str, output_path: str, columns_to_keep: list):
    # Create a SparkSession
    spark = SparkSession.builder \
        .appName("VectorSlicer") \
        .getOrCreate()

    # Load the data
    df = spark.read.parquet(input_path)

    # Define the VectorAssembler
    assembler = VectorAssembler(inputCols=columns_to_keep, outputCol="features")

    # Transform the data
    assembled_df = assembler.transform(df)

    # Define the indices for VectorSlicer based on the number of columns to keep
    indices = list(range(len(columns_to_keep)))

    # Define the VectorSlicer
    slicer = VectorSlicer(inputCol="features", outputCol="sliced_features", indices=indices)

    # Now we can transform our DataFrame
    output_df = slicer.transform(assembled_df)

    # Save the transformed DataFrame
    output_df.write.parquet(output_path)

# Columns to keep
columns_to_keep = ["UDI", "Air temperature [K]", "Process temperature [K]", 
                   "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", 
                   "TWF", "HDF", "PWF", "OSF", "RNF"]

# Call the function
process_data("/path/to/input", "/path/to/output", columns_to_keep)
