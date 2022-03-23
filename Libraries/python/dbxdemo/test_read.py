# Databricks notebook source
import pytest

from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils

class TestRead(object):
    def test_read(self):
        spark = SparkSession.builder.getOrCreate()

        path = "/user/sasaki_kohei@comture.com/hotel_booking"

        csvpath = "{}/hotel_bookings.csv".format(path)

        df = (spark.read
          .option("header", True)
          .option("inferSchema", True)
          .csv(csvpath))

        print (df.count())
        assert df.count() == 119390
