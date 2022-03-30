# Databricks notebook source
import pytest
import sasaki

class TestSasaki(object):
    def test_sasaki(self):
        check_column = ["hotel", "arrival_date_year", "arrival_date_month", "arrival_date_day_of_month", "count", "date", "MA_3DAY", "MA_7DAY"]
        
        df_pd = df.toPandas()
        resort_column = list(df_pd)

        for i in range(len(check_column)):
            assert (check_column[i] == resort_column[i]), "カラム名が一致しません。処理を中断します。"
        print("カラム名が一致しました。")
