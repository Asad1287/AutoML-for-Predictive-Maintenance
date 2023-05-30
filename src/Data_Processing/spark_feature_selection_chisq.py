from pyspark.ml.feature import ChiSqSelector, VectorAssembler
from pyspark.sql import SparkSession

def process_data(input_path: str, output_path: str, columns_to_keep: list):
    # Create a SparkSession
    spark = SparkSession.builder \
        .appName("ChiSqSelector") \
        .getOrCreate()

    # Load the data
    df = spark.read.parquet(input_path)

    # Define the VectorAssembler
    assembler = VectorAssembler(inputCols=columns_to_keep, outputCol="features")

    # Transform the data
    assembled_df = assembler.transform(df)

    # Define the ChiSqSelector
    # Here we select top 5 features showing the most important features 
    selector = ChiSqSelector(numTopFeatures=5, featuresCol="features",
                             outputCol="selectedFeatures", labelCol="label")

    # Apply the selector
    result = selector.fit(assembled_df).transform(assembled_df)

    # Save the transformed DataFrame
    result.write.parquet(output_path)

# Columns to keep
columns_to_keep = ["UDI", "Air temperature [K]", "Process temperature [K]", 
                   "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", 
                   "TWF", "HDF", "PWF", "OSF", "RNF"]

# Call the function
process_data("/path/to/input", "/path/to/output", columns_to_keep)
