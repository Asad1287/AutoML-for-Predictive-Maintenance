from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import *
import os

# Define the schema of the incoming JSON data
schema = StructType([
    StructField("Product ID", IntegerType(), True),
    # Add the rest of your fields here
])

# Initialize Spark session
spark = SparkSession.builder \
    .master("local") \
    .appName("kafka-spark") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0") \
    .getOrCreate()
# Read the incoming JSON data from Kafka
df = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "iot-data") \
    .load()

iot_data = df.select(from_json(col("value").cast("string"), schema).alias("iot_data")).select("iot_data.*")

# Write the deserialized data to CSV
query = iot_data \
    .writeStream \
    .outputMode("append") \
    .format("csv") \
    .option("path", "output/result.csv") \
    .option("checkpointLocation", "output/checkpoint") \
    .start()

# Wait for the query to finish
query.awaitTermination()