from pyspark.sql import SparkSession

spark = SparkSession.builder.master("local").appName("PySpark Test").getOrCreate()

data = [("Alice", 34), ("Bob", 45), ("Cathy", 29)]
columns = ["Name", "Age"]

df = spark.createDataFrame(data, columns)

df.show()

spark.stop()