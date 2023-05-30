from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import PipelineModel

def train_model():
    # Create Spark Session
    spark = SparkSession.builder.appName("RandomForestModel").getOrCreate()

    # Read from parquet file
    df = spark.read.parquet("data/processed_data.parquet")

    # Load pickle y_train
    y_train = pd.read_pickle("data/y_train.pkl")

    # Prepare data for Machine Learning
    col_names = df.columns
    col_names.remove('Machine failure') # Assuming 'Machine failure' is the target column
    assembler = VectorAssembler(inputCols=col_names, outputCol="features")

    # Initialize Random Forest Model
    rf = RandomForestClassifier(labelCol="Machine failure", featuresCol="features")

    # Construct pipeline
    pipeline = Pipeline(stages=[assembler, rf])

    # Set up the parameter grid
    paramGrid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [int(x) for x in range(20, 31)]) \
        .addGrid(rf.maxDepth, [int(x) for x in range(4, 7)]) \
        .build()

    # Establish the model evaluator
    evaluator = MulticlassClassificationEvaluator(
        labelCol="Machine failure", predictionCol="prediction", metricName="accuracy")

    # Create cross-validator
    crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=paramGrid,
                              evaluator=evaluator,
                              numFolds=3)

    # Fit Model
    cvModel = crossval.fit(df)

    # Save Model
    cvModel.save("model_rf_sparkml")
    print("Model saved.")

    # Load Model
    loaded_model = PipelineModel.load("model_rf_sparkml")
    print("Model loaded.")
    print(loaded_model)

    # Stop Spark Session
    spark.stop()

# Running the function
if __name__ == '__main__':
    train_model()