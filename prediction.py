import sys
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def preprocess_data(data_frame):
    """Prepares the dataset by converting columns to double and removing unnecessary quotes."""
    print("Preprocessing data: casting columns to double")
    return data_frame.select(*(col(column).cast("double").alias(column.strip("\"")) for column in data_frame.columns))

if __name__ == "__main__":
    print("Launching Spark Application on EMR Cluster: Wine_Cluster")

    # Initialize Spark session
    spark = SparkSession.builder.appName("WineQualityEvaluation").getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel('ERROR')

    # Configure S3 settings
    sc._jsc.hadoopConfiguration().set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")

    # Define S3 paths
    validation_dataset_path = str(sys.argv[1])  # Pass validation dataset path as an argument
    model_path = "s3://bucketcluster/trainedmodel"  # Updated S3 bucket path

    # Load validation dataset
    print(f"Loading validation data from: {validation_dataset_path}")
    raw_data = (spark.read
                .format("csv")
                .option('header', 'true')
                .option("sep", ";")
                .option("inferschema", 'true')
                .load(validation_dataset_path))

    # Preprocess data
    print("Processing the validation dataset")
    prepared_data = preprocess_data(raw_data)

    # Load trained model from S3
    print(f"Loading pre-trained model from: {model_path}")
    model = PipelineModel.load(model_path)

    # Generate predictions
    print("Running predictions on the validation data")
    predictions = model.transform(prepared_data)

    # Calculate evaluation metrics
    print("Evaluating predictions")
    prediction_output = predictions.select(['prediction', 'label'])
    evaluator = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='accuracy')
    accuracy_score = evaluator.evaluate(predictions)
    print(f'Accuracy of the wine quality prediction model = {accuracy_score}')

    # Compute F1 Score using RDD API
    print("Calculating F1 Score")
    prediction_rdd = prediction_output.rdd.map(tuple)
    metrics = MulticlassMetrics(prediction_rdd)
    f1_score = metrics.weightedFMeasure()
    print(f'Weighted F1 Score of the model = {f1_score}')

    # Stop Spark session
    print("Shutting down Spark Application")
    spark.stop()
