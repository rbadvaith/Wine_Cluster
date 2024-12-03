import sys
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def load_and_prepare_data(spark, data_path):
    """
    Loads and prepares the validation dataset.

    Args:
        spark: SparkSession object.
        data_path: Path to the validation dataset.

    Returns:
        A cleaned DataFrame ready for predictions.
    """
    print(f"Loading data from {data_path}")
    raw_data = (
        spark.read.format("csv")
        .option("header", "true")
        .option("sep", ";")
        .option("inferschema", "true")
        .load(data_path)
    )
    print("Cleaning and preparing data")
    cleaned_data = raw_data.select(
        *(col(column).cast("double").alias(column.strip("\"")) for column in raw_data.columns)
    )
    return cleaned_data


def evaluate_model(data, model_path):
    """
    Loads the trained model, makes predictions, and evaluates its performance.

    Args:
        data: The prepared validation DataFrame.
        model_path: Path to the trained model.

    Returns:
        None
    """
    print(f"Loading trained model from {model_path}")
    model = PipelineModel.load(model_path)

    print("Generating predictions on validation data")
    predictions = model.transform(data)

    print("Evaluating model performance")
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )
    accuracy = evaluator.evaluate(predictions)
    print(f"Test Accuracy of the model: {accuracy}")

    prediction_results = predictions.select(["prediction", "label"])
    metrics = MulticlassMetrics(prediction_results.rdd.map(tuple))
    f1_score = metrics.weightedFMeasure()
    print(f"Weighted F1 Score of the model: {f1_score}")


if __name__ == "__main__":
    print("Starting Spark application")

    spark = SparkSession.builder.appName("WineQualityPrediction").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # Set S3 configurations if necessary
    spark.sparkContext._jsc.hadoopConfiguration().set(
        "fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem"
    )

    # Paths to validation dataset and trained model
    validation_dataset_path = "ValidationDataset.csv"
    trained_model_path = "/job/trainedmodel"

    # Load and prepare validation data
    validation_data = load_and_prepare_data(spark, validation_dataset_path)

    # Evaluate the trained model on the validation data
    evaluate_model(validation_data, trained_model_path)

    print("Exiting Spark application")
    spark.stop()
