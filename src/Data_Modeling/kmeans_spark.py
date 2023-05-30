from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml import PipelineModel
import pandas as pd

def kmeans_model():
    # Create Spark Session
    spark = SparkSession.builder.appName("KMeansModel").getOrCreate()

    # Read from parquet file
    df = spark.read.parquet("data/processed_data.parquet")

    # Load pickle y_train
    y_train = pd.read_pickle("data/y_train.pkl")

    # Prepare data for Machine Learning
    col_names = df.columns
    col_names.remove('Machine failure')  # Assuming 'Machine failure' is the target column
    assembler = VectorAssembler(inputCols=col_names, outputCol="features")

    # Initialize KMeans Model
    kmeans = KMeans(featuresCol="features").setK(3).setSeed(1)  # Change number of clusters and seed as required

    # Construct pipeline
    pipeline = Pipeline(stages=[assembler, kmeans])

    # Fit Model
    model = pipeline.fit(df)

    # Make predictions
    predictions = model.transform(df)

    # Evaluate clustering by computing Silhouette score
    evaluator = ClusteringEvaluator()

    silhouette = evaluator.evaluate(predictions)
    print("Silhouette with squared euclidean distance = " + str(silhouette))

    # Shows the result.
    centers = model.stages[-1].clusterCenters()
    print("Cluster Centers: ")
    for center in centers:
        print(center)

    # Save Model
    model.save("model_kmeans_sparkml")
    print("Model saved.")

    # Load Model
    loaded_model = PipelineModel.load("model_kmeans_sparkml")
    print("Model loaded.")
    print(loaded_model)

    # Stop Spark Session
    spark.stop()

# Running the function
if __name__ == '__main__':
    kmeans_model()
