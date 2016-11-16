package org.upm.spark.flightdelayed

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark._
import org.apache.log4j.{Level, Logger}

object App {
  def main(args: Array[String]) {
    Logger.getRootLogger().setLevel(Level.WARN)
    val conf = new SparkConf().setAppName("My first Spark application")
    val sc = new SparkContext(conf)
    val data = sc.textFile("hdfs:///tmp/data/2008.csv")
    //println(data.count())
    data.take(20).foreach(println)
  }
}
