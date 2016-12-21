package org.upm.spark.flightdelayed

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, IntegerType}

object App {
  def main(args: Array[String]) {
    val inputPath = args {
      0
    }
    //val conf = new SparkConf().setAppName("flightDelayed")
    //val sc = new SparkContext(conf)

    //val data = sc.textFile("hdfs:///tmp/data/2008.csv")
    //val data = sc.textFile(inputPath)

    val sparkSQL = SparkSession.builder().enableHiveSupport().getOrCreate()

    val inputDataRaw1 = sparkSQL.read
      .format("com.databricks.spark.csv")
      .option("header", "true") //reading the headers
      .option("mode", "DROPMALFORMED")
      .option("inferSchema", "true")
      .option("delimiter", ";")
      .load(inputPath) //.csv("csv/file/path") //spark 2.0 api

    val inputDataRaw = inputDataRaw1.withColumn("target", inputDataRaw1("ArrDelay").cast("int"))

    val firstCleanRaw = inputDataRaw.where("Cancelled = 0")
    val firstCleanRaw2 = firstCleanRaw.where("ArrDelay != 'NA'")
    val firstClean = firstCleanRaw2.where("DepDelay != 'NA'")


    val dataClean = firstClean.drop(firstClean.col("ArrTime"))
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

    //DATA IS CONCENTRATED BETWEEN 5 minutes more or less. Skewness is very high!!


    def logMod(x: Integer): Double = {
      var res: Double = 0.0
      if (x >= 0) {
        var aux = math.abs(x)
        aux += 1
        val aux2 = math.log(aux)
        res = aux2
      } else {
        var aux = math.abs(x)
        aux += 1
        val aux2 = math.log(aux)
        res = -aux2
      }
      res
    }

    val logModUDF = udf(logMod _)

    val analysisData2 = analysisData
      .withColumn("DepDelay", logModUDF(analysisData("DepDelay")))
      .withColumn("target", logModUDF(analysisData("target")))
      .na.drop()


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

    val Array(training, test) = analysisData2.randomSplit(Array(0.7, 0.3))

    val model = tvs.fit(training)


    val holdout = model.transform(test).select("prediction", "target")


    val holdoutRes = holdout
      .withColumn("target", holdout("target").cast(DoubleType))
      .withColumn("prediction", holdout("prediction").cast(DoubleType))

    val rm = new RegressionMetrics(holdoutRes.rdd.map(x => (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double])))

    println("MSE: " + rm.meanSquaredError)
    println("R Squared: " + rm.r2)
    println("RMSE: "+rm.rootMeanSquaredError)
    println("Explained Variance: " + rm.explainedVariance + "\n")

  }
}
