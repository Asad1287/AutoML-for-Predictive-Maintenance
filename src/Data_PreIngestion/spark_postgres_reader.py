from pyspark.sql import SparkSession

def read_from_postgres_and_save_as_parquet(query, jdbc_url, output_path):
    # Create a SparkSession
    spark = SparkSession.builder.getOrCreate()

    # Read data from PostgreSQL
    df = spark.read.format("jdbc") \
        .option("url", jdbc_url) \
        .option("query", query) \
        .option("user", "your_username") \
        .option("password", "your_password") \
        .option("driver", "org.postgresql.Driver") \
        .load()

    # Save DataFrame as Parquet
    df.write.parquet(output_path)

# Usage:
query = "SELECT * FROM table_name"
jdbc_url = "jdbc:postgresql://localhost:5432/your_database"

read_from_postgres_and_save_as_parquet(query, jdbc_url, "/path/to/save/parquet/files")