from pyspark.sql import SparkSession

# Create a SparkSession
spark = SparkSession.builder \
    .appName("CSV to DataFrame") \
    .getOrCreate()

# Read the CSV file
df = spark.read.csv("ai4i2020.csv", header=True, inferSchema=True)

# Show the DataFrame
df.show()