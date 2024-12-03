import sys
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def initialize_spark(app_name):
    """Initializes the Spark session and configures S3 access."""
    print(f"Initializing Spark Session for {app_name}")
    spark = SparkSession.builder.appName(app_name).getOrCreate()
    spark_context = spark.sparkContext
    spark_context.setLogLevel('ERROR')
    spark_context._jsc.hadoopConfiguration().set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    return spark

def clean_data(data_frame):
    """Cleans data by casting columns to double and stripping extra quotes."""
    print("Cleaning data")
    return data_frame.select(*(col(column).cast("double").alias(column.strip("\"")) for column in data_frame.columns))

def create_pipeline(features_assembler, label_indexer, classifier):
    """Creates a Spark ML pipeline."""
    print("Creating training pipeline")
    return Pipeline(stages=[features_assembler, label_indexer, classifier])

def build_random_forest_classifier():
    """Builds a RandomForestClassifier with default parameters."""
    print("Building RandomForestClassifier")
    return RandomForestClassifier(labelCol='label', 
                                  featuresCol='features',
                                  numTrees=150,
                                  maxDepth=15,
                                  seed=150,
                                  impurity='gini')

def configure_cross_validator(pipeline, param_grid, evaluator):
    """Configures CrossValidator for hyperparameter tuning."""
    print("Configuring CrossValidator")
    return CrossValidator(estimator=pipeline,
                          estimatorParamMaps=param_grid,
                          evaluator=evaluator,
                          numFolds=2)

if __name__ == "__main__":
    print("Starting Spark Application on EMR Cluster: Wine_Cluster")

    # Initialize Spark session
    spark_session = initialize_spark("WineQualityPrediction")

    # Define S3 paths
    training_data_path = "s3://bucketcluster/TrainingDataset.csv"
    model_output_path = "s3://bucketcluster/trainedmodel"

    # Read training data
    print(f"Reading training data from {training_data_path}")
    raw_data_frame = (spark_session.read
                      .format("csv")
                      .option('header', 'true')
                      .option("sep", ";")
                      .option("inferschema", 'true')
                      .load(training_data_path))

    training_data_frame = clean_data(raw_data_frame)

    # Feature columns
    feature_columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                       'pH', 'sulphates', 'alcohol', 'quality']

    # Set up pipeline components
    print("Setting up feature assembler and label indexer")
    features_assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')
    label_indexer = StringIndexer(inputCol="quality", outputCol="label")

    # Cache data
    print("Caching training data")
    training_data_frame.cache()

    # Create classifier and pipeline
    random_forest_classifier = build_random_forest_classifier()
    training_pipeline = create_pipeline(features_assembler, label_indexer, random_forest_classifier)

    # Evaluate model
    accuracy_evaluator = MulticlassClassificationEvaluator(labelCol='label', 
                                                           predictionCol='prediction', 
                                                           metricName='accuracy')

    # Hyperparameter tuning
    print("Setting up parameter grid for CrossValidator")
    parameter_grid = ParamGridBuilder() \
        .addGrid(random_forest_classifier.maxDepth, [5, 10]) \
        .addGrid(random_forest_classifier.numTrees, [50, 150]) \
        .addGrid(random_forest_classifier.minInstancesPerNode, [5]) \
        .addGrid(random_forest_classifier.seed, [100, 200]) \
        .addGrid(random_forest_classifier.impurity, ["entropy", "gini"]) \
        .build()

    cross_validator = configure_cross_validator(training_pipeline, parameter_grid, accuracy_evaluator)

    # Train and tune model
    print("Fitting model with CrossValidator")
    best_model = cross_validator.fit(training_data_frame)
    final_model = best_model.bestModel

    # Save the model
    print(f"Saving the trained model to {model_output_path}")
    final_model.write().overwrite().save(model_output_path)

    print("Stopping Spark Session")
    spark_session.stop()
