import sys
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql import SparkSession
from pyspark.sql.functions import col


def prepare_data(spark, input_path):
    """
    Prepares and cleans the training data for machine learning.

    Args:
        spark: SparkSession object.
        input_path: Path to the training dataset.

    Returns:
        A cleaned and prepared DataFrame ready for training.
    """
    print(f"Loading data from {input_path}")
    raw_data = (
        spark.read.format("csv")
        .option("header", "true")
        .option("sep", ";")
        .option("inferschema", "true")
        .load(input_path)
    )
    print("Cleaning and formatting data")
    clean_data = raw_data.select(
        *(col(column).cast("double").alias(column.strip("\"")) for column in raw_data.columns)
    )
    return clean_data


def train_and_evaluate_model(data, model_path):
    """
    Trains a Random Forest classifier, performs hyperparameter tuning, and saves the best model.

    Args:
        data: Prepared training DataFrame.
        model_path: Path to save the trained model.
    """
    feature_columns = [
        'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
        'pH', 'sulphates', 'alcohol', 'quality'
    ]
    
    print("Setting up the pipeline components")
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    indexer = StringIndexer(inputCol="quality", outputCol="label")
    classifier = RandomForestClassifier(
        labelCol="label",
        featuresCol="features",
        numTrees=150,
        maxDepth=15,
        seed=150,
        impurity="gini"
    )
    
    pipeline = Pipeline(stages=[assembler, indexer, classifier])
    print("Fitting initial model pipeline")
    pipeline_model = pipeline.fit(data)

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )

    print("Building parameter grid for cross-validation")
    param_grid = ParamGridBuilder() \
        .addGrid(classifier.maxDepth, [5, 10]) \
        .addGrid(classifier.numTrees, [50, 150]) \
        .addGrid(classifier.minInstancesPerNode, [5]) \
        .addGrid(classifier.seed, [100, 200]) \
        .addGrid(classifier.impurity, ["entropy", "gini"]) \
        .build()
    
    print("Initializing CrossValidator")
    cross_validator = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=2
    )

    print("Training the model using cross-validation")
    best_model = cross_validator.fit(data).bestModel
    
    print(f"Saving the best model to {model_path}")
    best_model.write().overwrite().save(model_path)


if __name__ == "__main__":
    print("Starting Spark application")

    spark = SparkSession.builder.appName("WineQualityPrediction").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # Specify input and output paths
    input_path = "TrainingDataset.csv"
    model_save_path = "/job/trainedmodel"

    # Prepare data
    training_data = prepare_data(spark, input_path)

    # Cache data for faster processing
    print("Caching data for model training")
    training_data.cache()

    # Train model and save
    train_and_evaluate_model(training_data, model_save_path)

    print("Spark application completed")
    spark.stop()
