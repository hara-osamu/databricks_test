# Databricks notebook source
# spark自体の定義づけを行う(リモートアクセス用)
from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils
from IPython.core.display import display

def test_sasaki():
    spark = SparkSession.builder.getOrCreate()

    dbutils = DBUtils(spark)

    # COMMAND ----------

    # MAGIC %md
    # MAGIC サンプルデータをFileStoreにインポート。(実際にはAWSのファイルストレージから持ってくるみたいな処理がある？)<br>
    # MAGIC 使用するディレクトリを定義し、インポートしたファイルを移動させる。

    # COMMAND ----------

    path = "/user/sasaki_kohei@comture.com/hotel_booking"

    # COMMAND ----------

    #import os

    #if os.path.exists(path) == False:
    #    dbutils.fs.mkdirs(path)
    #    dbutils.fs.mv("/FileStore/hotel_bookings.csv", path)
    #else:
    #    print("directory is already exist.")

    # COMMAND ----------

    # MAGIC %md
    # MAGIC csvのデータをデータフレームに読み込む

    # COMMAND ----------

    csvpath = "{}/hotel_bookings.csv".format(path)

    df = (spark.read
      .option("header", True)
      .option("inferSchema", True)
      .csv(csvpath))

    # df.head()

    # COMMAND ----------

    # MAGIC %md
    # MAGIC データの概要を確認

    # COMMAND ----------

    # 2015/01 ~ 2017/12のデータが存在
    from pyspark.sql.functions import col, substring

    df1 = df.withColumn("arrival_date_month", substring(col("reservation_status_date"), 6, 2).cast("int"))
    #display(df1.select("arrival_date_year", "arrival_date_month").groupBy("arrival_date_year","arrival_date_month").count().sort("arrival_date_year", "arrival_date_month"))

    # COMMAND ----------

    # 2015年における、予約がキャンセルされなかった(実際に宿泊した)客数の月ごとの推移を確認
    # 加えて、Resort HotelとCity Hotelで推移の仕方に違いがあるか確認

    df1 = df.withColumn("arrival_date_month", substring(col("reservation_status_date"), 6, 2))
    df1 = df1.select("hotel", "is_canceled", "arrival_date_year", "arrival_date_month")
    df1 = df1.filter((col("is_canceled") == 0) & (col("arrival_date_year") == 2016)).groupBy("hotel", "arrival_date_month").count()
    # df1 = df1.cache()

    # COMMAND ----------

    # MAGIC %md 
    # MAGIC ### 何を予測する？
    # MAGIC ・2016年のデータをもとに2017年の宿泊予約数の予測を行う→移動平均モデル<br>
    # MAGIC ・客の予約キャンセル率を予測→機械学習の分類モデル

    # COMMAND ----------

    # MAGIC %md
    # MAGIC ■宿泊予約数の予測

    # COMMAND ----------

    # 月の英語表記を数字表記に変更する関数を定義
    from time import strptime

    def eng_to_num(month):
        num = strptime(month ,'%b').tm_mon
        return num

    # COMMAND ----------

    from pyspark.sql.functions import udf
    eng_to_numUDF = udf(eng_to_num)

    # COMMAND ----------

    # 2016年における、ホテルの形態(City or Resort)毎の1日当たりの来客数を集計
    from pyspark.sql.functions import col, substring, concat

    df_2016 = (df.filter((col("is_canceled") == 0) & (col("arrival_date_year") == 2016))
             .select("hotel", "arrival_date_year", eng_to_numUDF(substring(col("arrival_date_month"), 1, 3)).cast("int").alias("arrival_date_month"), "arrival_date_day_of_month")
             .groupBy("hotel", "arrival_date_year", "arrival_date_month", "arrival_date_day_of_month").count()
             .sort("arrival_date_year", "arrival_date_month", "arrival_date_day_of_month")
          )

    df_2016 = df_2016.filter((col("arrival_date_month") < 9))
    #df_2016.head()

    # COMMAND ----------

    # うるう年の影響を排除
    df_2016 = df_2016.filter(~((col("arrival_date_month") == 2) & (col("arrival_date_day_of_month") == 29)))
    #df_2016.count()

    # COMMAND ----------

    # データフレームをpandas型に変換
    resort_df = df_2016.filter(col("hotel") == "Resort Hotel")
    city_df = df_2016.filter(col("hotel") == "City Hotel")

    # COMMAND ----------

    resort = resort_df.toPandas()

    # COMMAND ----------

    city = city_df.toPandas()

    # COMMAND ----------

    import pandas as pd

    # COMMAND ----------

    # 年月日を一緒にした日付列を作成
    resort["date"] = resort["arrival_date_year"].apply(str) + "/" + resort["arrival_date_month"].apply(str) + "/" + resort["arrival_date_day_of_month"].apply(str)
    city["date"] = city["arrival_date_year"].apply(str) + "/" + city["arrival_date_month"].apply(str) + "/" + city["arrival_date_day_of_month"].apply(str)

    resort["date"] = pd.to_datetime(resort["date"], format="%Y/%m/%d")
    city["date"] = pd.to_datetime(city["date"], format="%Y/%m/%d")

    # COMMAND ----------

    # 移動平均の計算
    resort["MA_3DAY"] = resort["count"].rolling(3).mean().round(1)
    resort["MA_7DAY"] = resort["count"].rolling(7).mean().round(1)


    city["MA_3DAY"] = city["count"].rolling(3).mean().round(1)
    city["MA_7DAY"] = city["count"].rolling(7).mean().round(1)

    # COMMAND ----------

    print(resort.shape)

    # COMMAND ----------

    print(city.shape)

    # COMMAND ----------

    # リゾートホテルの予約数の移動平均線をプロット
    import matplotlib.pyplot as plt

    plt.xlabel("Date")
    plt.ylabel("Count")

    plt.plot(resort["date"], resort["MA_3DAY"], label="MA_3DAY", color="red")
    plt.plot(resort["date"], resort["MA_7DAY"], label="MA_7DAY", color="blue")
    plt.legend(loc="best")

    plt.show()

    # COMMAND ----------

    # 2017年における、ホテルの形態(City or Resort)毎の1日当たりの来客数を集計
    from pyspark.sql.functions import col, substring, concat

    df_2017 = (df.filter((col("is_canceled") == 0) & (col("arrival_date_year") == 2017))
             .select("hotel", "arrival_date_year", eng_to_numUDF(substring(col("arrival_date_month"), 1, 3)).cast("int").alias("arrival_date_month"), "arrival_date_day_of_month")
             .groupBy("hotel", "arrival_date_year", "arrival_date_month", "arrival_date_day_of_month").count()
             .sort("arrival_date_year", "arrival_date_month", "arrival_date_day_of_month")
          )
    # display(df_2017)

    # COMMAND ----------

    # データフレームをpandas型に変換
    resort_2017 = df_2017.filter(col("hotel") == "Resort Hotel").toPandas()
    city_2017 = df_2017.filter(col("hotel") == "City Hotel").toPandas()

    # COMMAND ----------

    # 年月日を一緒にした日付列を作成
    resort_2017["date"] = resort_2017["arrival_date_year"].apply(str) + "/" + resort_2017["arrival_date_month"].apply(str) + "/" + resort_2017["arrival_date_day_of_month"].apply(str)
    city_2017["date"] = city_2017["arrival_date_year"].apply(str) + "/" + city_2017["arrival_date_month"].apply(str) + "/" + city_2017["arrival_date_day_of_month"].apply(str)

    resort_2017["date"] = pd.to_datetime(resort_2017["date"], format="%Y/%m/%d")
    city_2017["date"] = pd.to_datetime(city_2017["date"], format="%Y/%m/%d")

    # COMMAND ----------

    # リゾートホテルにおける、2016年の移動平均線と2017年の実値を比較
    import matplotlib.pyplot as plt

    plt.xlabel("Date")
    plt.ylabel("Count")

    plt.plot(resort_2017["date"], resort["MA_3DAY"], label="MA_3DAY", color="red")
    plt.plot(resort_2017["date"], resort_2017["count"], label="real_value", color="blue")
    plt.legend(loc="best")

    plt.show()

    # COMMAND ----------

    # MAGIC %md
    # MAGIC テスト用プログラム部分

    # COMMAND ----------

    #実行結果を別にCSVファイルに残しておく
    return_resort = spark.createDataFrame(resort)

    outputpath = "dbfs:/user/sasaki_kohei@comture.com/output"
    resort_path = outputpath + "/resort"

    dbutils.fs.rm(resort_path, True)

    # '_started'と'_committed_'で始まるファイルを書き込まないように設定
    spark.conf.set("spark.sql.sources.commitProtocolClass", "org.apache.spark.sql.execution.datasources.SQLHadoopMapReduceCommitProtocol")

    # '_SUCCESS'で始まるファイルを書き込まないように設定
    spark.conf.set("mapreduce.fileoutputcommitter.marksuccessfuljobs","false")

    (return_resort.coalesce(1)
         .write
         .format("csv")
         .option("header", True)
         .mode("overwrite")
         .save(resort_path)
    )

    # COMMAND ----------

    from py4j.java_gateway import java_import
    from pyspark import SparkContext
    sc = SparkContext.getOrCreate();

    java_import(spark._jvm, "org.apache.hadoop.fs.Path");

    fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
    file = fs.globStatus(sc._jvm.Path('/user/sasaki_kohei@comture.com/output/resort/*'))[0].getPath().getName()
    fs.rename(sc._jvm.Path('/user/sasaki_kohei@comture.com/output/resort/' + file), sc._jvm.Path('/user/sasaki_kohei@comture.com/output/resort/resort.csv'))
    # fs.delete(sc._jvm.Path('/user/sasaki_kohei@comture.com/output/resort/'), True)
    display(dbutils.fs.ls(resort_path))

    # COMMAND ----------

    #import glob
    #import os

    #file_path_for_python = f'{resort_path}/*'.replace("dbfs:","/dbfs")
    #print(file_path_for_python)

    #file_base_name = "resort"
    #file_extension = ".csv"

    # ファイル一覧の作成
    #file_list = glob.glob(file_path_for_python)
    #print(file_list)

    # ファイル一覧の名称を変更
    #for i, file_path in enumerate(file_list):
    #    file_dir = os.path.dirname(file_path)
    #    print(file_dir)

    #    new_file_path =os.path.join(file_dir,
    #                           file_base_name + 
    #                           file_extension)
    #    os.rename(file_path, new_file_path)

    #display(dbutils.fs.ls(resort_path))

    # COMMAND ----------

    # dbutils.notebook.run("unit_test", 60)
    # result = dbutils.notebook.run("/Users/sasaki_kohei@comture.com/Demand-Forecast/unit_test", 60)
    # print(json.loads(result))

    # COMMAND ----------

    # MAGIC %md
    # MAGIC 
    # MAGIC ### 結果
    # MAGIC 客数の増減に関しては移動平均線の動き通りであったが、増加と減少の幅が2017年の方が大きく、予測値とは大幅に異なっていることが分かる。<br>
    # MAGIC →客数の予測モデルとしては説明力に欠けるものと思われる。

