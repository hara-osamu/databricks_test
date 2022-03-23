# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils

spark = SparkSession.builder.getOrCreate()

path = "/user/sasaki_kohei@comture.com/hotel_booking"

csvpath = "{}/hotel_bookings.csv".format(path)

df = (spark.read
  .option("header", True)
  .option("inferSchema", True)
  .csv(csvpath))

print (df.count())
