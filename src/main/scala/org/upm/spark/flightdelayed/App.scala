package org.upm.spark.flightdelayed

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, OneHotEncoder}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.evaluation.{RegressionEvaluator}
import org.apache.spark.ml.regression.{LinearRegression}
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.types.DoubleType
import org.apache.log4j.{Level, Logger}

object App {
  def main(args: Array[String]) {
    Logger.getRootLogger().setLevel(Level.WARN)
    val inputPath = args {
      0
    }
    val conf = new SparkConf().setAppName("flightDelayed")
    val sc = new SparkContext(conf)

    //val data = sc.textFile("hdfs:///tmp/data/2008.csv")
    //val data = sc.textFile(inputPath)

    val sparkSQL = SparkSession.builder().enableHiveSupport().getOrCreate()

    val inputData = sparkSQL.read
      .format("com.databricks.spark.csv")
      .option("header", "true") //reading the headers
      .option("mode", "DROPMALFORMED")
      .option("inferSchema", "true")
      .option("delimiter", ";")
      .load("/FileStore/tables/okqu8mhe1481897675184/test3.csv") //.csv("csv/file/path") //spark 2.0 api
      .withColumn("target", $"ArrDelay".cast("int"))

    val clean = inputData.where("ArrDelay is not null and ArrDelay != 'NA'")
    val firstClean = clean.where("Cancelled = 0")


    val dataClean = firstClean.drop(inputData.col("ArrTime"))
      .drop(firstClean.col("ActualElapsedTime"))
      .drop(firstClean.col("AirTime"))
      .drop(firstClean.col("TaxiIn"))
      .drop(firstClean.col("Diverted"))
      .drop(firstClean.col("CarrierDelay"))
      .drop(firstClean.col("WeatherDelay"))
      .drop(firstClean.col("NASDelay"))
      .drop(firstClean.col("SecurityDelay"))
      .drop(firstClean.col("LateAircraftDelay"))
      .drop(firstClean.col("Cancelled"))
      .drop(firstClean.col("CancellationCode"))
      .drop()

    val analysisData = dataClean
      .withColumn("DepDelay", dataClean("DepDelay").cast(IntegerType))
      .withColumn("TaxiOut", dataClean("TaxiOut").cast(IntegerType))
      .withColumn("CRSElapsedTime", dataClean("CRSElapsedTime").cast(IntegerType))
      .withColumn("Month", dataClean("Month").cast(IntegerType))
      .withColumn("DayofMonth", dataClean("DayofMonth").cast(IntegerType))
      .withColumn("DayOfWeek", dataClean("DayOfWeek").cast(IntegerType))
      .withColumn("DepTime", dataClean("DepTime").cast(IntegerType))
      .withColumn("CRSDepTime", dataClean("CRSDepTime").cast(IntegerType))
      .withColumn("CRSArrTime", dataClean("CRSArrTime").cast(IntegerType))
      .withColumn("FlightNum", dataClean("FlightNum").cast(IntegerType))
      .withColumn("ArrDelay", dataClean("ArrDelay").cast(IntegerType))
      .withColumn("Distance", dataClean("Distance").cast(IntegerType))
      .withColumn("target", dataClean("target").cast(IntegerType))


    val categoricalVariables = Array("UniqueCarrier")
    val categoricalIndexers = categoricalVariables
      .map(i => new StringIndexer()
        .setInputCol(i)
        .setOutputCol(i + "Index"))
    val categoricalEncoders = categoricalVariables
      .map(e => new OneHotEncoder()
        .setInputCol(e + "Index")
        .setOutputCol(e + "Vec"))


    val assembler = new VectorAssembler()
      .setInputCols(Array("UniqueCarrierVec", "DepDelay", "TaxiOut", "CRSElapsedTime"))
      .setOutputCol("features")


    val lr = new LinearRegression()
      .setLabelCol("target")
      .setFeaturesCol("features")
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.1, 0.01))
      .addGrid(lr.fitIntercept)
      .addGrid(lr.elasticNetParam, Array(0.0, 1.0))
      .build()

    val steps: Array[org.apache.spark.ml.PipelineStage] = categoricalIndexers ++ categoricalEncoders ++ Array(assembler, lr)
    val pipeline = new Pipeline().setStages(steps)

    val tvs = new TrainValidationSplit()
      .setEstimator(pipeline) // the estimator can also just be an individual model rather than a pipeline
      .setEvaluator(new RegressionEvaluator().setLabelCol("target"))
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.75)

    val cleanAnalysisData = analysisData.where("Cancelled = 0 and ArrDelay is not null")

    val Array(training, test) = analysisData.randomSplit(Array(0.7, 0.3))


    val model = tvs.fit(training)


    val holdout = model.transform(test).select("prediction", "target")

    var holdoutRes = holdout
      .withColumn("target", holdout("target").cast(DoubleType))
      .withColumn("prediction", holdout("prediction").cast(DoubleType))

    val rm = new RegressionMetrics(holdoutRes.rdd.map(x => (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double])))

    println("MSE: " + rm.meanSquaredError)
    println("R Squared: " + rm.r2)
    println("Explained Variance: " + rm.explainedVariance + "\n")

    //TODO: Remove this variables:
    //● ArrTime
    //● ActualElapsedTime
    //● AirTime
    //● TaxiIn
    //● Diverted
    //● CarrierDelay
    //● WeatherDelay
    //● NASDelay
    //● SecurityDelay
    //● LateAircraftDelay

    //TODO: This are the most significant variables:
    //DepDelay
    //TaxiOut
    //CRSElapsedTime

    //TODO: THIS IS REGRESSION MODEL

    //TRANSFORMERS: A Transformer is an abstraction that includes feature transformers and learned models.
    //A Transformer implements a method transform(), which converts one DataFrame into another, generally by appending one or more columns.
    //FOR EXAMPLE:
    //A feature transformer might take a DataFrame, read a column (e.g., text), map it into a new column (e.g., feature vectors), and output a new DataFrame with the mapped column appended.
    //A learning model might take a DataFrame, read the column containing feature vectors, predict the label for each feature vector, and output a new DataFrame with predicted labels appended as a column.

    //ESTIMATOR: An Estimator abstracts the concept of a learning algorithm or any algorithm that fits or trains on data.
    //An Estimator implements a method fit(), which accepts a DataFrame and produces a Model, which is a Transformer.
    //FOR EXAMPLE:
    //A learning algorithm such as LogisticRegression is an Estimator, and calling fit() trains a LogisticRegressionModel , which is a Model and hence a Transformer

    //PIPELINE: In machine learning, it is common to run a sequence of algorithms to process and learn from data.
    //Example: A simple text document processing workflow might include several stages:
    //Split each document’s text into words.
    //Convert each document’s words into a numerical feature vector.
    //Learn a prediction model using the feature vectors and labels.
    //MLlib represents such a workflow as a Pipeline, which consists of a sequence of PipelineStages (Transformers and Estimators) to be run in a specific order
    //A Pipeline is specified as a sequence of stages (Transformer or Estimator)
    //Stages are run in order
    //The input DataFrame is transformed as it passes through each stage
    //For Transformer stages, the transform() method is called on the DataFrame
    //For Estimator stages, the fit() method is called to produce a Transformer, and that Transformer’s transform() method is called on the DataFrame
    //A Pipeline is an Estimator
    //After a Pipeline’s fit() method runs, it produces a PipelineModel, which is a Transformer
    //This PipelineModel is used at test time


  }
}
