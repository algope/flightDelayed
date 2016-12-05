package org.upm.spark.flightdelayed

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.apache.log4j.{Level, Logger}

object App {
  def main(args: Array[String]) {
    Logger.getRootLogger().setLevel(Level.WARN)
    val inputPath = args{0}
    val conf = new SparkConf().setAppName("My first Spark application")
    val sc = new SparkContext(conf)

    //val data = sc.textFile("hdfs:///tmp/data/2008.csv")
    //val data = sc.textFile(inputPath)

    val sparkSQL = SparkSession.builder().enableHiveSupport().getOrCreate()

    val inputData = sparkSQL.read.csv(inputPath)
    inputData.printSchema()
    inputData.show()

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

    val split = inputData.randomSplit(Array(0.7,0.3)) //70% for training 30% for test


  }
}
