from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType
PASSWORD = "1234"
INFERENCE_COLLECTION = "inference_results"
DATABASE = "predictive_maintenance"

col_names = ['UDI', 'Product ID', 'Type', 'Air temperature',
       'Process temperature', 'Rotational speed', 'Torque',
       'Tool wear', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF', 'Machine failure']

PASSWORD = "1234"
INFERENCE_COLLECTION = "inference_results"
DATABASE = "predictive_maintenance"

mongo_uri = f"mongodb+srv://root12345:{PASSWORD}@cluster1.b03tix4.mongodb.net/{DATABASE}.{INFERENCE_COLLECTION}"

# Create SparkSession
spark = SparkSession \
    .builder \
    .appName("Spark parquet write") \
    .config("spark.mongodb.output.uri", mongo_uri) \
    .config("spark.jars", "/mnt/d/Portfolio/Auto_ML_Pdm/AutoML/FullCode/mongo-spark-connector_2.12-3.0.0.jar")\
    .getOrCreate()

# Load data from Parquet file
df = spark.read.parquet("data/inference_results.parquet")

columns = df.columns

  # Convert each column to double type.
for column in columns:
    df = df.withColumn(column, col(column).cast(DoubleType()))

# Write data to MongoDB
df.write.format("com.mongodb.spark.sql.DefaultSource").mode("overwrite").save()

# Close the SparkSession
spark.stop()

