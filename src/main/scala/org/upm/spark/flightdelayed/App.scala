package org.upm.spark.flightdelayed

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.apache.spark._
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

  }
}
