# Databricks notebook source
# MAGIC %md ##Estimating Future Spend
# MAGIC 
# MAGIC ##将来の支出を見積もる
# MAGIC 
# MAGIC In the previous notebook, we examined how customers in a non-subscription model disengage over time.  Without formal contracts in place between retailers and customers, we are left to estimate the probability a customer has dropped out of an on-going relationship based on their historical patterns of engagement relative to those of other customers. Understanding the probability a customer remains actively engaged is highly valuable in its own right.  But we can take it one step further and calculate how much revenue or profit we might derive from this predicted future engagement. 
# MAGIC 
# MAGIC 前回のノートでは、非定額課金モデルの顧客が時間の経過とともにどのように離脱していくかについて検討しました。小売業者と顧客の間に正式な契約がない場合、他の顧客と比較した過去のエンゲージメントパターンに基づいて、顧客が継続的な関係から脱落する確率を推定することになる。顧客が積極的に関与し続ける確率を理解することは、それ自体で非常に価値がある。しかし、もう一歩踏み込んで、この予測された将来のエンゲージメントからどれだけの収益や利益が得られるかを計算することができる。
# MAGIC 
# MAGIC To do this, we must build a model calculating the monetary value associated with future purchase events.  The purpose of this model is to derive such a model and combine it with lifetime probabilities to derive estimated Customer Lifetime Value.
# MAGIC 
# MAGIC そのためには、将来の購入イベントに関連する金銭的価値を計算するモデルを構築する必要があります。このモデルの目的は、このようなモデルを導き出し、それをライフタイム確率と組み合わせて、推定顧客ライフタイムバリューを導き出すことである。

# COMMAND ----------

# MAGIC %md ###Step 1: Setup the Environment
# MAGIC 
# MAGIC ###ステップ1：環境の構築
# MAGIC As before, you need to attach this notebook to a cluster running the **Databricks ML runtime** version 6.5 or higher with the following libraries [installed](https://docs.databricks.com/libraries.html#workspace-libraries) and [attached](https://docs.databricks.com/libraries.html#install-a-library-on-a-cluster):</p>
# MAGIC 
# MAGIC 前回同様、Databricks MLランタイムバージョン6.5以上が動作するクラスタに、以下のライブラリをインストールし、このノートブックを添付する必要があります。
# MAGIC 
# MAGIC * xlrd
# MAGIC * lifetimes==0.10.1
# MAGIC * nbconvert
# MAGIC 
# MAGIC In addition, you need to load the [Online Retail Data Set](http://archive.ics.uci.edu/ml/datasets/Online+Retail) available from the UCI Machine Learning Repository to the */FileStore/tables/online_retail/* folder in your environment, as described in the previous notebook. With prerequisites in place, we can access the dataset as follows:
# MAGIC 
# MAGIC さらに、前回のノートブックで説明したように、UCI Machine Learning Repositoryから利用できるOnline Retail Data Setを環境の/FileStore/tables/online_retail/フォルダにロードしておく必要があります。前提条件が整ったことで、以下のようにデータセットにアクセスできるようになりました。

# COMMAND ----------

import pandas as pd
import numpy as np

# identify name of xlsx file (which can change when uploaded)
xlsx_filename = dbutils.fs.ls('file:///dbfs/FileStore/tables/online_retail')[0][0]

# schema of the excel spreadsheet data range
orders_schema = {
  'InvoiceNo':np.str,
  'StockCode':np.str,
  'Description':np.str,
  'Quantity':np.int64,
  'InvoiceDate':np.datetime64,
  'UnitPrice':np.float64,
  'CustomerID':np.str,
  'Country':np.str  
  }

# read spreadsheet to pandas dataframe
# the xlrd library (loaded at the top of this notebook) is required for this step to work 
orders_pd = pd.read_excel(
  xlsx_filename, 
  sheet_name='Online Retail',
  header=0, # first row is header
  dtype=orders_schema
  )

# COMMAND ----------

# MAGIC %md In order to examine the monetary value associated with customer purchases, we need to calculate the amount of each sale in the online retail orders data set. This is done by simply multiplying Quantity by UnitPrice to create a new SalesAmount field:
# MAGIC 
# MAGIC 顧客の購入に関連する金銭的価値を調べるには、オンライン小売注文データセットの各販売額を計算する必要があります。これは、単に Quantity に UnitPrice を掛けて、新しい SalesAmount フィールドを作成することで行われます。

# COMMAND ----------

# calculate sales amount as quantity * unit price
orders_pd['SalesAmount'] = orders_pd['Quantity'] * orders_pd['UnitPrice']

orders_pd.head(10)

# COMMAND ----------

# MAGIC %md Let's now make this data available to Spark for later use:
# MAGIC 
# MAGIC では、このデータをSparkで利用できるようにしましょう。

# COMMAND ----------

# convert pandas DF to Spark DF
orders = spark.createDataFrame(orders_pd)

# present Spark DF as queriable view
orders.createOrReplaceTempView('orders') 

# COMMAND ----------

# MAGIC %md ###Step 2: Explore the Dataset
# MAGIC 
# MAGIC ###ステップ2：データセットの探索
# MAGIC 
# MAGIC For an examination of the purchase frequency patterns in the dataset, we'll refer you to the Step 2 section of the prior notebook.  Here, we wish to examine patterns related to customer spend.  
# MAGIC 
# MAGIC データセットの購入頻度パターンを調べるには、前のノートのステップ2のセクションを参照してください。ここでは、顧客の支出に関連するパターンを調べたいと思います。
# MAGIC 
# MAGIC To get started, let's take a look at the typical daily spend of a customer.  We will group this at a daily level because like with the customer lifetime calculations, we will consider multiple purchases on the same day to be part of the same purchase event:
# MAGIC 
# MAGIC 手始めに、顧客の1日の典型的な支出を見てみましょう。顧客のライフタイム計算と同様に、同じ日の複数の購入を同じ購入イベントの一部と見なすため、これを1日単位でグループ化します。

# COMMAND ----------

# MAGIC %sql -- daily sales by customer
# MAGIC 
# MAGIC SELECT
# MAGIC   CustomerID,
# MAGIC   TO_DATE(InvoiceDate) as InvoiceDate,
# MAGIC   SUM(SalesAmount) as SalesAmount
# MAGIC FROM orders
# MAGIC GROUP BY CustomerID, TO_DATE(InvoiceDate)

# COMMAND ----------

# MAGIC %md The range of daily spend is quite wide with a few customers purchasing over £70,000 in a single day.  Without much knowledge of the underlying business, it's hard to say if this is level of spending is consistent with the expectations of the site or an outlier which should be removed.  
# MAGIC 
# MAGIC 1日の消費額の幅はかなり広く、1日に7万ポンド以上購入する顧客も数人いる。この支出レベルが、サイトの期待値と一致しているのか、それとも削除すべき異常値なのかは、基本的なビジネスについてあまり知識がないため、判断がつきません。
# MAGIC 
# MAGIC Notice too that there are quite a few negative values which are most likely associated with returns.  We'll discuss this more later in this notebook, but for now, we'll narrow the range of values we are examining to get a sense of the distribution of the bulk of the activity observed on the site:
# MAGIC 
# MAGIC また、マイナスの値も多く、これは返品に関連する可能性が高いことに注意してください。この点については後ほど詳しく説明しますが、ここでは、サイト上で観察されるアクティビティの大半の分布の感覚をつかむために、調査する値の範囲を狭めることにします。

# COMMAND ----------

# MAGIC %sql -- daily sales by customer (for daily sales between 0 and 2500£)
# MAGIC 
# MAGIC SELECT
# MAGIC   CustomerID,
# MAGIC   TO_DATE(InvoiceDate) as InvoiceDate,
# MAGIC   SUM(SalesAmount) as SalesAmount
# MAGIC FROM orders
# MAGIC GROUP BY CustomerID, TO_DATE(InvoiceDate)
# MAGIC HAVING SalesAmount BETWEEN 0 AND 2500

# COMMAND ----------

# MAGIC %md The distribution of daily spend in this narrowed range is centered around 200 to 400 pound sterling with a long-tail towards higher ranges of spend. It's clear this is not a normal (gaussian) distribution.
# MAGIC 
# MAGIC この絞り込まれた範囲での1日の消費額の分布は、200から400ポンドを中心とし、それ以上の範囲の消費額に向かってロングテールとなっています。これは、明らかに正規分布（ガウス分布）ではない。
# MAGIC 
# MAGIC This same distribution pattern is observed in the spending patterns of individual customers. Focusing on a few customers with a high number of purchases, you can see that spending patterns vary but this right-skewed pattern persists:
# MAGIC 
# MAGIC この分布パターンは、個々の顧客の支出パターンでも同じように観察されます。購入回数の多い数人の顧客に焦点を当てると、支出パターンは様々ですが、この右肩上がりのパターンが持続していることが分かります。

# COMMAND ----------

# MAGIC %sql -- top customers by frequency
# MAGIC 
# MAGIC SELECT
# MAGIC   CustomerID,
# MAGIC   COUNT(DISTINCT TO_DATE(InvoiceDate)) as Frequency
# MAGIC FROM orders
# MAGIC GROUP BY CustomerID
# MAGIC ORDER BY Frequency DESC
# MAGIC LIMIT 5

# COMMAND ----------

# MAGIC %sql -- daily spend for three most frequent customers
# MAGIC 
# MAGIC SELECT
# MAGIC   CustomerID,
# MAGIC   TO_DATE(InvoiceDate) as InvoiceDate,
# MAGIC   SUM(SalesAmount) as SalesAmount
# MAGIC FROM orders
# MAGIC WHERE CustomerID IN (14911, 12748, 17841)
# MAGIC GROUP BY CustomerID, TO_DATE(InvoiceDate)
# MAGIC ORDER BY CustomerID

# COMMAND ----------

# MAGIC %md There's a bit more we need to examine in this dataset but first we must calculate some per-customer metrics.
# MAGIC 
# MAGIC このデータセットにはもう少し検討すべき点がありますが、まずは顧客ごとの指標を算出する必要があります。

# COMMAND ----------

# MAGIC %md ###Step 3: Calculate Customer Metrics
# MAGIC 
# MAGIC ###ステップ3：顧客指標を算出する
# MAGIC The dataset with which we are working contains raw transactional history.  As before, we need to calculate the per-customer metrics of frequency, age (T), and recency but we also need to calculate a monetary value metric:</p>
# MAGIC 
# MAGIC 今回扱うデータセットには、生の取引履歴が含まれている。前と同様に、頻度、年齢 (T)、再帰性という顧客ごとの指標を計算する必要があるが、金額的な指標も計算する必要がある。
# MAGIC 
# MAGIC * **Frequency** - the number of dates on which a customer made a purchase subsequent to the date of the customer's first purchase.
# MAGIC * **Age (T)** - the number of time units, *e.g.* days, since the date of a customer's first purchase to the current date (or last date in the dataset).
# MAGIC * **Recency** - the age of the customer (as previously defined) at the time of their last purchase.
# MAGIC * **Monetary Value** - the average per transaction-date spend by a customer during repeat purchases.  (Margin and other monetary values may also be used if available.)
# MAGIC 
# MAGIC * 頻度 - 顧客が最初に購入した日以降に購入を行った日付の数。
# MAGIC * 年齢 (T) - 顧客が最初に購入した日から現在の日付 (またはデータセットの最後の日付) までの、日数などの時間単位の数。
# MAGIC * 再来店 - 顧客の最後の購入時の年齢(前に定義した通り)。
# MAGIC * Monetary Value - 顧客が繰り返し購入する際の、取引日ごとの平均消費額。(マージンや他の貨幣価値も利用可能であれば使用されるかもしれない)。
# MAGIC 
# MAGIC It's important to note that when calculating metrics such as customer age that we need to consider when our dataset terminates.  Calculating these metrics relative to today's date can lead to erroneous results.  Given this, we will identify the last date in the dataset and define that as *today's date* for all calculations.
# MAGIC 
# MAGIC 顧客年齢のような指標を計算するときに、データセットがいつ終了するかを考慮する必要があることに注意することが重要である。今日の日付を基準にしてこれらのメトリクスを計算すると、誤った結果になることがあります。そこで、データセットの最後の日付を特定し、それをすべての計算のための今日の日付として定義します。
# MAGIC 
# MAGIC To derive these metrics, we may make use of some built-in functionality in the [lifetimes](https://lifetimes.readthedocs.io/en/latest/lifetimes.html) library.  If you've reviewed the code in the prior notebook, you may recognize the method being called is identical to the one used before.  The only difference is that we are instructing the method to use our SalesAmount field as our measure of monetary value:
# MAGIC 
# MAGIC これらのメトリクスを導き出すために、lifetimesライブラリのいくつかの組み込み機能を利用することができます。前のノートブックでコードを確認した方は、呼び出されるメソッドが以前使用したものと同じであることにお気づきかもしれません。唯一の違いは、SalesAmountフィールドを金銭的価値の指標として使用するようにメソッドに指示している点です。

# COMMAND ----------

import lifetimes

# set the last transaction date as the end point for this historical dataset
current_date = orders_pd['InvoiceDate'].max()

# calculate the required customer metrics
metrics_pd = (
  lifetimes.utils.summary_data_from_transaction_data(
    orders_pd,
    customer_id_col='CustomerID',
    datetime_col='InvoiceDate',
    observation_period_end = current_date, 
    freq='D',
    monetary_value_col='SalesAmount'  # use sales amount to determine monetary value
    )
  )

# display first few rows
metrics_pd.head(10)

# COMMAND ----------

# MAGIC %md As before, let's examine how we might generate this same dataset using Spark so that when working with larger datasets, we can calculate these values in a parallelized manner.  Just like before, the logic in the next two cells demonstrates how we might do this using first a SQL statement and then the Programmatic SQL API.  We've worked to keep the code as consistent with the prior notebook whenever possible with the exception of the additional logic required for the monetary value logic:
# MAGIC 
# MAGIC 前回と同様に、Sparkを使って同じデータセットを生成し、より大きなデータセットを扱うときに、並列化された方法でこれらの値を計算できるようにする方法を検討しましょう。前回と同様に、次の2つのセルのロジックは、最初にSQLステートメントを使用し、次にProgrammatic SQL APIを使用して、どのようにこれを行うかを示しています。金額ロジックに必要な追加ロジックを除き、可能な限り以前のノートブックと一貫したコードを維持するように努めました。

# COMMAND ----------

# sql statement to derive summary customer stats
sql = '''
  SELECT
    a.customerid as CustomerID,
    CAST(COUNT(DISTINCT a.transaction_at) - 1 as float) as frequency,
    CAST(DATEDIFF(MAX(a.transaction_at), a.first_at) as float) as recency,
    CAST(DATEDIFF(a.current_dt, a.first_at) as float) as T,
    CASE                                              -- MONETARY VALUE CALCULATION
      WHEN COUNT(DISTINCT a.transaction_at)=1 THEN 0    -- 0 if only one order
      ELSE
        SUM(
          CASE WHEN a.first_at=a.transaction_at THEN 0  -- daily average of all but first order
          ELSE a.salesamount
          END
          ) / (COUNT(DISTINCT a.transaction_at)-1)
      END as monetary_value    
  FROM ( -- customer order history
    SELECT
      x.customerid,
      z.first_at,
      x.transaction_at,
      y.current_dt,
      x.salesamount                  
    FROM (                                            -- customer daily summary
      SELECT 
        customerid, 
        TO_DATE(invoicedate) as transaction_at, 
        SUM(SalesAmount) as salesamount               -- SALES AMOUNT ADDED
      FROM orders 
      GROUP BY customerid, TO_DATE(invoicedate)
      ) x
    CROSS JOIN (SELECT MAX(TO_DATE(invoicedate)) as current_dt FROM orders) y                                -- current date (according to dataset)
    INNER JOIN (SELECT customerid, MIN(TO_DATE(invoicedate)) as first_at FROM orders GROUP BY customerid) z  -- first order per customer
      ON x.customerid=z.customerid
    WHERE x.customerid IS NOT NULL
    ) a
  GROUP BY a.customerid, a.current_dt, a.first_at
  ORDER BY CustomerID
  '''

# capture stats in dataframe 
metrics_sql = spark.sql(sql)

# display stats
display(metrics_sql)  

# COMMAND ----------

# programmatic sql api calls to derive summary customer stats

from pyspark.sql.functions import to_date, datediff, max, min, countDistinct, count, sum, when
from pyspark.sql.types import *

# valid customer orders
x = (
    orders
      .where(orders.CustomerID.isNotNull())
      .withColumn('transaction_at', to_date(orders.InvoiceDate))
      .groupBy(orders.CustomerID, 'transaction_at')
      .agg(sum(orders.SalesAmount).alias('salesamount'))   # SALES AMOUNT
    )

# calculate last date in dataset
y = (
  orders
    .groupBy()
    .agg(max(to_date(orders.InvoiceDate)).alias('current_dt'))
  )

# calculate first transaction date by customer
z = (
  orders
    .groupBy(orders.CustomerID)
    .agg(min(to_date(orders.InvoiceDate)).alias('first_at'))
  )

# combine customer history with date info 
a = (x
    .crossJoin(y)
    .join(z, x.CustomerID==z.CustomerID, how='inner')
    .select(
      x.CustomerID.alias('customerid'), 
      z.first_at, 
      x.transaction_at,
      x.salesamount,               # SALES AMOUNT
      y.current_dt
      )
    )

# calculate relevant metrics by customer
metrics_api = (a
           .groupBy(a.customerid, a.current_dt, a.first_at)
           .agg(
             (countDistinct(a.transaction_at)-1).cast(FloatType()).alias('frequency'),
             datediff(max(a.transaction_at), a.first_at).cast(FloatType()).alias('recency'),
             datediff(a.current_dt, a.first_at).cast(FloatType()).alias('T'),
             when(countDistinct(a.transaction_at)==1,0)                           # MONETARY VALUE
               .otherwise(
                 sum(
                   when(a.first_at==a.transaction_at,0)
                     .otherwise(a.salesamount)
                   )/(countDistinct(a.transaction_at)-1)
                 ).alias('monetary_value')
               )
           .select('customerid','frequency','recency','T','monetary_value')
           .orderBy('customerid')
          )

display(metrics_api)

# COMMAND ----------

# MAGIC %md And as before, we can use some summary stats to verify the datasets generated via SQL are identical to those generated by the lifetimes library:
# MAGIC 
# MAGIC また、前回と同様に、SQLで生成されたデータセットがlifetimesライブラリで生成されたものと同一であることを、いくつかの要約統計を使って確認することができます。

# COMMAND ----------

# summary data from lifetimes
metrics_pd.describe()

# COMMAND ----------

# summary data from SQL statement
metrics_sql.toPandas().describe()

# COMMAND ----------

# summary data from pyspark.sql API
metrics_api.toPandas().describe()

# COMMAND ----------

# MAGIC %md Expanding these calculations to derive values for calibration and holdout periods, the logic is as follows:
# MAGIC 
# MAGIC この計算を発展させて、キャリブレーション期間とホールドアウト期間の値を導き出すと、以下のようなロジックになります。
# MAGIC 
# MAGIC NOTE Again, we are using a widget to define the number of days in the holdout period.
# MAGIC 
# MAGIC 注：ここでも、ホールドアウト期間の日数を定義するためにウィジェットを使用しています。

# COMMAND ----------

# define a notebook parameter making holdout days configurable (90-days default)
dbutils.widgets.text('holdout days', '90')

# COMMAND ----------

from datetime import timedelta

# set the last transaction date as the end point for this historical dataset
current_date = orders_pd['InvoiceDate'].max()

# define end of calibration period
holdout_days = int(dbutils.widgets.get('holdout days'))
calibration_end_date = current_date - timedelta(days = holdout_days)

# calculate the required customer metrics
metrics_cal_pd = (
  lifetimes.utils.calibration_and_holdout_data(
    orders_pd,
    customer_id_col='CustomerID',
    datetime_col='InvoiceDate',
    observation_period_end = current_date,
    calibration_period_end=calibration_end_date,
    freq='D',
    monetary_value_col='SalesAmount'  # use sales amount to determine monetary value
    )
  )

# display first few rows
metrics_cal_pd.head(10)

# COMMAND ----------

# MAGIC %md Implementing the required logic in SQL and using the Programmatic SQL API, we arrive at the following:
# MAGIC 
# MAGIC 必要なロジックをSQLで実装し、Programmatic SQL APIを使用すると、次のようになります。
# MAGIC 
# MAGIC ※原追記　widgetの仕様が変わったのか、getArgumentが文字扱いされてエラーを返してきていたためcastで対応

# COMMAND ----------

sql = '''
WITH CustomerHistory 
  AS (
    SELECT  -- nesting req'ed b/c can't SELECT DISTINCT on widget parameter
      m.*,
      cast(getArgument('holdout days') as int) as duration_holdout
    FROM (
      SELECT
        x.customerid,
        z.first_at,
        x.transaction_at,
        y.current_dt,
        x.salesamount
      FROM (                                            -- CUSTOMER DAILY SUMMARY
        SELECT 
          customerid, 
          TO_DATE(invoicedate) as transaction_at, 
          SUM(SalesAmount) as salesamount 
        FROM orders 
        GROUP BY customerid, TO_DATE(invoicedate)
        ) x
      CROSS JOIN (SELECT MAX(TO_DATE(invoicedate)) as current_dt FROM orders) y                                -- current date (according to dataset)
      INNER JOIN (SELECT customerid, MIN(TO_DATE(invoicedate)) as first_at FROM orders GROUP BY customerid) z  -- first order per customer
        ON x.customerid=z.customerid
      WHERE x.customerid is not null
      ) m
  )
SELECT
    a.customerid as CustomerID,
    a.frequency as frequency_cal,
    a.recency as recency_cal,
    a.T as T_cal,
    COALESCE(a.monetary_value,0.0) as monetary_value_cal,
    COALESCE(b.frequency_holdout, 0.0) as frequency_holdout,
    COALESCE(b.monetary_value_holdout, 0.0) as monetary_value_holdout,
    a.duration_holdout
FROM ( -- CALIBRATION PERIOD CALCULATIONS
    SELECT
        p.customerid,
        CAST(p.duration_holdout as float) as duration_holdout,
        CAST(DATEDIFF(MAX(p.transaction_at), p.first_at) as float) as recency,
        CAST(COUNT(DISTINCT p.transaction_at) - 1 as float) as frequency,
        CAST(DATEDIFF(DATE_SUB(p.current_dt, p.duration_holdout), p.first_at) as float) as T,
        CASE                                              -- MONETARY VALUE CALCULATION
          WHEN COUNT(DISTINCT p.transaction_at)=1 THEN 0    -- 0 if only one order
          ELSE
            SUM(
              CASE WHEN p.first_at=p.transaction_at THEN 0  -- daily average of all but first order
              ELSE p.salesamount
              END
              ) / (COUNT(DISTINCT p.transaction_at)-1)
          END as monetary_value    
    FROM CustomerHistory p
    WHERE p.transaction_at < DATE_SUB(p.current_dt, p.duration_holdout)  -- LIMIT THIS QUERY TO DATA IN THE CALIBRATION PERIOD
    GROUP BY p.customerid, p.duration_holdout, p.current_dt, p.first_at
  ) a
LEFT OUTER JOIN ( -- HOLDOUT PERIOD CALCULATIONS
  SELECT
    p.customerid,
    CAST(COUNT(DISTINCT p.transaction_at) as float) as frequency_holdout,
    AVG(p.salesamount) as monetary_value_holdout      -- MONETARY VALUE CALCULATION
  FROM CustomerHistory p
  WHERE 
    p.transaction_at >= DATE_SUB(p.current_dt, p.duration_holdout) AND  -- LIMIT THIS QUERY TO DATA IN THE HOLDOUT PERIOD
    p.transaction_at <= p.current_dt
  GROUP BY p.customerid
  ) b
  ON a.customerid=b.customerid
ORDER BY CustomerID
'''

metrics_cal_sql = spark.sql(sql)
display(metrics_cal_sql)

# COMMAND ----------

from pyspark.sql.functions import avg, date_sub, coalesce, lit, expr

# valid customer orders
x = (
  orders
    .where(orders.CustomerID.isNotNull())
    .withColumn('transaction_at', to_date(orders.InvoiceDate))
    .groupBy(orders.CustomerID, 'transaction_at')
    .agg(sum(orders.SalesAmount).alias('salesamount'))
  )

# calculate last date in dataset
y = (
  orders
    .groupBy()
    .agg(max(to_date(orders.InvoiceDate)).alias('current_dt'))
  )

# calculate first transaction date by customer
z = (
  orders
    .groupBy(orders.CustomerID)
    .agg(min(to_date(orders.InvoiceDate)).alias('first_at'))
  )

# combine customer history with date info (CUSTOMER HISTORY)
p = (x
    .crossJoin(y)
    .join(z, x.CustomerID==z.CustomerID, how='inner')
    .withColumn('duration_holdout', lit(int(dbutils.widgets.get('holdout days'))))
    .select(
      x.CustomerID.alias('customerid'),
      z.first_at, 
      x.transaction_at, 
      y.current_dt, 
      x.salesamount,
      'duration_holdout'
      )
     .distinct()
    )

# calculate relevant metrics by customer
# note: date_sub requires a single integer value unless employed within an expr() call
a = (p
       .where(p.transaction_at < expr('date_sub(current_dt, duration_holdout)')) 
       .groupBy(p.customerid, p.current_dt, p.duration_holdout, p.first_at)
       .agg(
         (countDistinct(p.transaction_at)-1).cast(FloatType()).alias('frequency_cal'),
         datediff( max(p.transaction_at), p.first_at).cast(FloatType()).alias('recency_cal'),
         datediff( expr('date_sub(current_dt, duration_holdout)'), p.first_at).cast(FloatType()).alias('T_cal'),
         when(countDistinct(p.transaction_at)==1,0)
           .otherwise(
             sum(
               when(p.first_at==p.transaction_at,0)
                 .otherwise(p.salesamount)
               )/(countDistinct(p.transaction_at)-1)
             ).alias('monetary_value_cal')
       )
    )

b = (p
      .where((p.transaction_at >= expr('date_sub(current_dt, duration_holdout)')) & (p.transaction_at <= p.current_dt) )
      .groupBy(p.customerid)
      .agg(
        countDistinct(p.transaction_at).cast(FloatType()).alias('frequency_holdout'),
        avg(p.salesamount).alias('monetary_value_holdout')
        )
   )

metrics_cal_api = (
                 a
                 .join(b, a.customerid==b.customerid, how='left')
                 .select(
                   a.customerid.alias('CustomerID'),
                   a.frequency_cal,
                   a.recency_cal,
                   a.T_cal,
                   a.monetary_value_cal,
                   coalesce(b.frequency_holdout, lit(0.0)).alias('frequency_holdout'),
                   coalesce(b.monetary_value_holdout, lit(0.0)).alias('monetary_value_holdout'),
                   a.duration_holdout
                   )
                 .orderBy('CustomerID')
              )

display(metrics_cal_api)

# COMMAND ----------

# MAGIC %md And now we compare the results:
# MAGIC 
# MAGIC そして、今度はその結果を比較します。

# COMMAND ----------

# summary data from lifetimes
metrics_cal_pd.describe()

# COMMAND ----------

# summary data from SQL statement
metrics_cal_sql.toPandas().describe()

# COMMAND ----------

# summary data from pyspark.sql API
metrics_cal_api.toPandas().describe()

# COMMAND ----------

# MAGIC %md Carefully examine the monetary holdout value calculated with the lifetimes library.  You should notice the values produced are significantly lower than those arrived at by the Spark code.  This is because the lifetimes library is averaging the individual line items on a given transaction date instead of averaging the transaction date total.  A change request has been submitted with the caretakers of the lifetimes library, but we believe the average of transaction date totals is the correct value and will use that for the remainder of this notebook.
# MAGIC 
# MAGIC lifetimesライブラリで計算された金額的なホールドアウト値を注意深く調べてみてください。Sparkコードで計算された値よりかなり低い値になっていることに気づくはずです。これは、ライフタイムライブラリが、取引日の合計を平均するのではなく、取引日の個々の項目を平均しているためです。ライフタイムライブラリの管理者に変更要求が提出されましたが、我々は取引日の合計の平均が正しい値であると考え、このノートブックの残りの部分はこれを使用することにします。
# MAGIC 
# MAGIC If you'd like to examine how values identical to those currently produced by the lifetimes library can be created using Spark, we've recreated the lifetimes logic using SQL here and provided a summary comparison in the two cells that follow:
# MAGIC 
# MAGIC 現在lifetimesライブラリで生成されている値と同じ値がSparkでどのように生成されるかを検証したい場合は、ここでSQLを使ってlifetimesロジックを再現し、以下の2つのセルでまとめて比較しています。
# MAGIC 
# MAGIC ※原追記　widgetの仕様が変わったのか、getArgumentが文字扱いされてエラーを返してきていたためcastで対応

# COMMAND ----------

sql = '''
WITH CustomerHistory 
  AS (
    SELECT  -- nesting req'ed b/c can't SELECT DISTINCT on widget parameter
      m.*,
      cast(getArgument('holdout days') as int) as duration_holdout
    FROM (
      SELECT
        x.customerid,
        z.first_at,
        x.transaction_at,
        y.current_dt,
        x.salesamount
      FROM (                                            -- CUSTOMER DAILY SUMMARY
        SELECT 
          customerid, 
          TO_DATE(invoicedate) as transaction_at, 
          SUM(SalesAmount) as salesamount 
        FROM orders 
        GROUP BY customerid, TO_DATE(invoicedate)
        ) x
      CROSS JOIN (SELECT MAX(TO_DATE(invoicedate)) as current_dt FROM orders) y                                -- current date (according to dataset)
      INNER JOIN (SELECT customerid, MIN(TO_DATE(invoicedate)) as first_at FROM orders GROUP BY customerid) z  -- first order per customer
        ON x.customerid=z.customerid
      WHERE x.customerid is not null
      ) m
  )
SELECT
    a.customerid as CustomerID,
    a.frequency as frequency_cal,
    a.recency as recency_cal,
    a.T as T_cal,
    COALESCE(a.monetary_value,0.0) as monetary_value_cal,
    COALESCE(b.frequency_holdout, 0.0) as frequency_holdout,
    COALESCE(b.monetary_value_holdout, 0.0) as monetary_value_holdout,
    a.duration_holdout
FROM ( -- CALIBRATION PERIOD CALCULATIONS
    SELECT
        p.customerid,
        CAST(p.duration_holdout as float) as duration_holdout,
        CAST(DATEDIFF(MAX(p.transaction_at), p.first_at) as float) as recency,
        CAST(COUNT(DISTINCT p.transaction_at) - 1 as float) as frequency,
        CAST(DATEDIFF(DATE_SUB(p.current_dt, p.duration_holdout), p.first_at) as float) as T,
        CASE                                              -- MONETARY VALUE CALCULATION
          WHEN COUNT(DISTINCT p.transaction_at)=1 THEN 0    -- 0 if only one order
          ELSE
            SUM(
              CASE WHEN p.first_at=p.transaction_at THEN 0  -- daily average of all but first order
              ELSE p.salesamount
              END
              ) / (COUNT(DISTINCT p.transaction_at)-1)
          END as monetary_value    
    FROM CustomerHistory p
    WHERE p.transaction_at < DATE_SUB(p.current_dt, p.duration_holdout)  -- LIMIT THIS QUERY TO DATA IN THE CALIBRATION PERIOD
    GROUP BY p.customerid, p.duration_holdout, p.current_dt, p.first_at
  ) a
LEFT OUTER JOIN ( -- HOLDOUT PERIOD CALCULATIONS
  SELECT
    p.customerid,
    CAST(COUNT(DISTINCT TO_DATE(p.invoicedate)) as float) as frequency_holdout,
    AVG(p.salesamount) as monetary_value_holdout      -- MONETARY VALUE CALCULATION
  FROM orders p
  CROSS JOIN (SELECT MAX(TO_DATE(invoicedate)) as current_dt FROM orders) q                                -- current date (according to dataset)
  INNER JOIN (SELECT customerid, MIN(TO_DATE(invoicedate)) as first_at FROM orders GROUP BY customerid) r  -- first order per customer
    ON p.customerid=r.customerid
  WHERE 
    p.customerid is not null AND
    TO_DATE(p.invoicedate) >= DATE_SUB(q.current_dt, cast(getArgument('holdout days') as int)) AND  -- LIMIT THIS QUERY TO DATA IN THE HOLDOUT PERIOD
    TO_DATE(p.invoicedate) <= q.current_dt
  GROUP BY p.customerid
  ) b
  ON a.customerid=b.customerid
ORDER BY CustomerID
'''

metrics_cal_sql_alt = spark.sql(sql)
display(metrics_cal_sql_alt)

# COMMAND ----------

# summary data from lifetimes
metrics_cal_pd.describe()

# COMMAND ----------

# summary data from alternative sql statement recreating
# monetary_value_holdout as currently implemented in lifetimes
metrics_cal_sql_alt.toPandas().describe()

# COMMAND ----------

# MAGIC %md Moving forward, we now need to limit our analysis to those customers with repeat purchases, just as we did in the prior notebook:
# MAGIC 
# MAGIC 今後は、先ほどのノートと同様に、リピート購入されるお客様に限定して分析する必要があります。

# COMMAND ----------

# remove customers with no repeats (complete dataset)
filtered = metrics_api.where(metrics_api.frequency > 0)

# remove customers with no repeats in calibration period
filtered_cal = metrics_cal_api.where(metrics_cal_api.frequency_cal > 0)

# COMMAND ----------

# MAGIC %md Finally, we need to consider what to do about the negative daily totals found in our dataset.  Without any contextual information about the retailer from which this dataset is derived, we might assume these negative values represent returns.  Ideally, we'd match returns to their original purchases and adjust the monetary values for the original transaction date.  That said, we do not have the information required to consistently do this and so we will simply include the negative return values in our daily transaction totals. Where this causes a daily total to be £0 or lower, we will simply exclude that value from our analysis.  Outside of a demonstration setting, this would typically be inappropriate:
# MAGIC 
# MAGIC 最後に、今回のデータセットで見つかった一日の合計がマイナスであることについて、どうするか検討する必要がある。このデータセットの元となった小売業者に関する文脈的な情報がない場合、これらのマイナスの値は返品を意味すると考えられるかもしれません。理想的には、返品を元の購入商品と照合し、元の取引日に合わせて金額を調整することです。しかし、私たちは一貫してこれを行うために必要な情報を持っていないため、単純にマイナスの返品を日々の取引合計に含めます。このため、1 日の合計が £0 以下になる場合は、その値を分析から除外することになります。デモの場以外では、これは通常、不適切です。

# COMMAND ----------

# exclude dates with negative totals (see note above) 
filtered = filtered.where(filtered.monetary_value > 0)
filtered_cal = filtered_cal.where(filtered_cal.monetary_value_cal > 0)

# COMMAND ----------

# MAGIC %md ###Step 4: Verifying Frequency & Monetary Value Independence
# MAGIC 
# MAGIC ###ステップ4：頻度と貨幣価値の独立性の検証
# MAGIC Before proceeding with our modeling, the gamma-gamma model (named for the two gamma distributions described earlier) that we will employ assumes that the frequency of a customer's purchases does not affect the monetary value of those purchases.  It's important that we test this and we can do so with the calculation of a simple Pearson's coefficient against our frequency and monetary value metrics.  We'll do this for the entire dataset, ignoring the calibration and holdout subsets, for this one analysis:
# MAGIC 
# MAGIC モデリングを進める前に、これから採用するガンマ・ガンマモデル（前述の2つのガンマ分布から命名）は、顧客の購入頻度がそれらの購入の金額に影響しないことを仮定しています。これをテストすることは重要で、頻度と貨幣価値の測定基準に対する単純なピアソンの係数を計算することでそれを行うことができます。この分析のために、キャリブレーションとホールドアウトのサブセットを無視して、データセット全体についてこれを行います。

# COMMAND ----------

filtered.corr('frequency', 'monetary_value')

# COMMAND ----------

# MAGIC %md While not perfectly independent, the correlation between these two values is pretty low so that we should be able to proceed with model training.
# MAGIC 
# MAGIC 完全に独立しているわけではありませんが、この2つの値の相関はかなり低いので、モデルの学習を進めることができると思われます。

# COMMAND ----------

# MAGIC %md ###Step 5: Train the Spend Model
# MAGIC 
# MAGIC ###ステップ5: 支出モデルのトレーニング
# MAGIC 
# MAGIC With our metrics in place, we can now train a model to estimate the monetary value to be derived from a future transactional event. The model we will use is referred to as the [Gamma-Gamma model](http://www.brucehardie.com/notes/025/gamma_gamma.pdf) in that it fits the gamma-distribution of an individual customer's spend against a gamma-distributed parameter that's derived from the customer population's spending distribution. The math is complex but the implementation is pretty straightforward using the lifetimes library.
# MAGIC 
# MAGIC 測定基準が整ったので、将来のトランザクション・イベントから得られる金銭的価値を推定するためのモデルをトレーニングすることができます。私たちが使うモデルは、ガンマ-ガンマモデルと呼ばれ、顧客集団の支出分布から得られるガンマ分布のパラメータに対して、個々の顧客の支出のガンマ分布を当てはめるというものです。この計算は複雑ですが、lifetimesライブラリを使用した実装は非常に簡単です。
# MAGIC 
# MAGIC That said, we must first determine the best value for the L2-regularization parameter used by the model.  For this, we will return to [hyperopt](http://hyperopt.github.io/hyperopt/), leveraging near identical patterns explored in the previous notebook:
# MAGIC 
# MAGIC とはいえ、まず、このモデルで使用するL2正則化パラメータの最適値を決定する必要があります。このために、我々はhyperoptに戻り、前のノートブックで調べたのとほぼ同じパターンを活用することにします。

# COMMAND ----------

from hyperopt import hp, fmin, tpe, rand, SparkTrials, STATUS_OK, space_eval

from lifetimes.fitters.gamma_gamma_fitter import GammaGammaFitter

# define search space
search_space = hp.uniform('l2', 0.0, 1.0)

# evaluation function
def score_model(actuals, predicted, metric='mse'):
  # make sure metric name is lower case
  metric = metric.lower()
  
  # Mean Squared Error and Root Mean Squared Error
  if metric=='mse' or metric=='rmse':
    val = np.sum(np.square(actuals-predicted))/actuals.shape[0]
    if metric=='rmse':
        val = np.sqrt(val)
  
  # Mean Absolute Error
  elif metric=='mae':
    np.sum(np.abs(actuals-predicted))/actuals.shape[0]
  
  else:
    val = None
  
  return val

# define function for model training and evaluation
def evaluate_model(param):
  
  # accesss replicated input_pd dataframe
  data = inputs.value
  
  # retrieve incoming parameters
  l2_reg = param
  
  # instantiate and configure the model
  model = GammaGammaFitter(penalizer_coef=l2_reg)
  
  # fit the model
  model.fit(data['frequency_cal'], data['monetary_value_cal'])
  
  # evaluate the model
  monetary_actual = data['monetary_value_holdout']
  monetary_predicted = model.conditional_expected_average_profit(data['frequency_holdout'], data['monetary_value_holdout'])
  mse = score_model(monetary_actual, monetary_predicted, 'mse')
  
  # return score and status
  return {'loss': mse, 'status': STATUS_OK}

# COMMAND ----------

# configure hyperopt settings to distribute to all executors on workers
spark_trials = SparkTrials(parallelism=2)

# select optimization algorithm
algo = tpe.suggest

# replicate input_pd dataframe to workers in Spark cluster
input_pd = filtered_cal.where(filtered_cal.monetary_value_cal > 0).toPandas()
inputs = sc.broadcast(input_pd)

# perform hyperparameter tuning (logging iterations to mlflow)
argmin = fmin(
  fn=evaluate_model,
  space=search_space,
  algo=algo,
  max_evals=100,
  trials=spark_trials
  )

# release the broadcast dataset
inputs.unpersist()

# COMMAND ----------

# print optimum hyperparameter settings
print(space_eval(search_space, argmin))

# COMMAND ----------

# MAGIC %md With our optimal L2 value identified, let's train the final spend model:
# MAGIC 
# MAGIC 最適なL2値を特定した後、最終的な支出モデルをトレーニングしましょう。

# COMMAND ----------

# get hyperparameter setting
l2_reg = space_eval(search_space, argmin)

# instantiate and configure model
spend_model = GammaGammaFitter(penalizer_coef=l2_reg)

# fit the model
spend_model.fit(input_pd['frequency_cal'], input_pd['monetary_value_cal'])

# COMMAND ----------

# MAGIC %md ###Step 6: Evaluate the Spend Model
# MAGIC 
# MAGIC ###ステップ6: 支出モデルの評価
# MAGIC 
# MAGIC The evaluation of the spend model is fairly straightforward.  We might examine how well predicted values align with actuals in the holdout period and derive an MSE from it:
# MAGIC 
# MAGIC 支出モデルの評価は、かなり簡単です。予測値とホールドアウト期間の実績がどの程度一致しているかを調べ、そこからMSEを導き出すことができます。

# COMMAND ----------

# evaluate the model
monetary_actual = input_pd['monetary_value_holdout']
monetary_predicted = spend_model.conditional_expected_average_profit(input_pd['frequency_holdout'], input_pd['monetary_value_holdout'])
mse = score_model(monetary_actual, monetary_predicted, 'mse')

print('MSE: {0}'.format(mse))

# COMMAND ----------

# MAGIC %md We might also visually inspect how are predicted spend values align with actuals, a technique employed in the [original paper](http://www.brucehardie.com/notes/025/gamma_gamma.pdf) that described the Gamma-Gamma model:
# MAGIC 
# MAGIC また、予測された支出額が実際の支出額とどのように一致しているかを視覚的に確認することもできます。これは、ガンマガンマモデルを説明したオリジナルの論文で採用された手法です。

# COMMAND ----------

import matplotlib.pyplot as plt

# define histogram bin count
bins = 10

# plot size
plt.figure(figsize=(15, 5))

# histogram plot values and presentation
plt.hist(monetary_actual, bins, label='actual', histtype='bar', color='STEELBLUE', rwidth=0.99)
plt.hist( monetary_predicted, bins, label='predict', histtype='step', color='ORANGE',  rwidth=0.99)

# place legend on chart
plt.legend(loc='upper right')

# COMMAND ----------

# MAGIC %md With only 10 bins, our model looks like it lines up with our actuals data pretty nicely.  If we expand the bin count, we see that the model underpredicts the occurrence of the lowest valued spend while following the remaining structure of the data.  Interestingly, a similar pattern was observed in the original paper cited earlier:
# MAGIC 
# MAGIC 10ビンのみで、モデルと実際のデータはうまく一致しているように見えます。ビンの数を増やすと、モデルはデータの残りの構造に沿いつつ、最も値の低いspendの発生を過小に予測することがわかります。興味深いことに、同じようなパターンが、先に引用したオリジナルの論文でも観察されています。

# COMMAND ----------

# define histogram bin count
bins = 40

# plot size
plt.figure(figsize=(15, 5))

# histogram plot values and presentation
plt.hist(monetary_actual, bins, label='actual', histtype='bar', color='STEELBLUE', rwidth=0.99)
plt.hist( monetary_predicted, bins, label='predict', histtype='step', color='ORANGE',  rwidth=0.99)

# place legend on chart
plt.legend(loc='upper right')

# COMMAND ----------

# MAGIC %md ###Step 7: Calculate Customer Lifetime Value
# MAGIC 
# MAGIC ###ステップ7：顧客生涯価値（Customer Lifetime Value）の算出
# MAGIC 
# MAGIC The spend model allows us to calculate the monetary value we may obtain from future purchase events.  When used in combination with the lifetime model which calculates the probable count of future spending events, we can derive a Customer Lifetime Value for a future period of time.
# MAGIC 
# MAGIC 消費モデルにより、将来の購入イベントから得られるであろう金銭的価値を計算することができます。将来の消費イベントの発生確率を計算するライフタイムモデルと組み合わせて使用することで、将来の一定期間の顧客生涯価値を導き出すことができます。
# MAGIC 
# MAGIC To demonstrate this, we must first train a lifetime model.  We'll use a BG/NBD model with an L2 parameter setting derived in an earlier execution of the prior notebook for this:
# MAGIC 
# MAGIC これを実証するために、まずライフタイムモデルをトレーニングする必要があります。ここでは、以前のノートブックの実行で導き出されたL2パラメータ設定のBG/NBDモデルを使用することにします。

# COMMAND ----------

from lifetimes.fitters.beta_geo_fitter import BetaGeoFitter

lifetime_input_pd = filtered_cal.toPandas() # pull data into pandas from Spark dataframe

# instantiate & configure the model (provided settings from previous hyperparam tuning exercise)
lifetimes_model = BetaGeoFitter(penalizer_coef=0.9995179967263891)

# train the model
lifetimes_model.fit(lifetime_input_pd['frequency_cal'], lifetime_input_pd['recency_cal'], lifetime_input_pd['T_cal'])

# score the model
frequency_holdout_actual = lifetime_input_pd['frequency_holdout']
frequency_holdout_predicted = lifetimes_model.predict(lifetime_input_pd['duration_holdout'], lifetime_input_pd['frequency_cal'], lifetime_input_pd['recency_cal'], lifetime_input_pd['T_cal'])
mse = score_model(frequency_holdout_actual, frequency_holdout_predicted, 'mse')

print('MSE: {0}'.format(mse))

# COMMAND ----------

# MAGIC %md Now we can use these together to calculate CLV. Here, we will calculate CLV over a 12-month period using a monthly discount rate of 1%:
# MAGIC 
# MAGIC さて、これらを組み合わせてCLVを計算します。ここでは、毎月の割引率を1％として、12ヶ月間のCLVを計算する。
# MAGIC 
# MAGIC NOTE The CFO typically defines a discount rate which should be used for these types of calculations.  Be sure the discount rate is expressed as a monthly discount rate.  If provided an annual discount rate, be sure to convert it to monthly using [this formula](https://www.experiglot.com/2006/06/07/how-to-convert-from-an-annual-rate-to-an-effective-periodic-rate-javascript-calculator/).
# MAGIC 
# MAGIC 注：CFOは通常、この種の計算に使用すべき割引率を定義している。割引率が月次割引率であることを確認すること。年間割引率が提供されている場合は、必ずこの式で月割りに変換してください。

# COMMAND ----------

clv_input_pd = filtered.toPandas()

# calculate the 1-year CLV for each customer
clv_input_pd['clv'] = (
  spend_model.customer_lifetime_value(
    lifetimes_model, #the model to use to predict the number of future transactions
    clv_input_pd['frequency'],
    clv_input_pd['recency'],
    clv_input_pd['T'],
    clv_input_pd['monetary_value'],
    time=12, # months
    discount_rate=0.01 # monthly discount rate ~ 12.7% annually
  )
)

clv_input_pd.head(10)

# COMMAND ----------

# MAGIC %md CLV is a powerful metric used by organizations to plan targeted promotional activities and assess customer equity. As such, it would be very helpful if we could convert our models into an easy to use function which we could employ in batch, streaming and interactive scenarios.
# MAGIC 
# MAGIC CLVは、企業がターゲットを絞ったプロモーション活動を計画し、カスタマーエクイティを評価するために使用される強力な指標である。そのため、私たちのモデルを、バッチ、ストリーミング、およびインタラクティブなシナリオで使用できる使いやすい関数に変換できれば、非常に便利です。
# MAGIC 
# MAGIC If you reviewed the prior notebook, you know where we are headed.  The one wrinkle we need to address here is that the CLV calculation depends on two models, not one.  Not a problem.  What we'll do is simply save the lifetime model as a pickled artifact associated with our spend model and in the custom wrapper we'll develop for our spend model, we'll re-instantiate the lifetime model so that it is available for predictions.
# MAGIC 
# MAGIC 前回のノートをご覧になれば、私たちがどこに向かっているかはおわかりいただけると思います。CLVの計算は、1つのモデルではなく、2つのモデルに依存していることが、ここで対処すべき1つの問題です。問題ありません。ここでは、ライフタイムモデルを支出モデルに関連付けられたピクルス状のアーティファクトとして保存し、支出モデル用に開発するカスタムラッパーで、ライフタイムモデルを再インスタンスして、予測に利用できるようにします。
# MAGIC 
# MAGIC To get started, let's save our lifetime model to a temporary location:
# MAGIC 
# MAGIC まず、ライフタイムモデルを一時的な場所に保存します。

# COMMAND ----------

# location to save temp copy of lifetimes model
lifetimes_model_path = '/dbfs/tmp/lifetimes_model.pkl'

# delete any prior copies that may exist
try:
  dbutils.fs.rm(lifetimes_model_path)
except:
  pass

# save the model to the temp location
lifetimes_model.save_model(lifetimes_model_path)

# COMMAND ----------

# MAGIC %md Now, let's define the custom wrapper for our spend model.  Notice that the *predict()* method is fairly simple and returns just a CLV value.  Notice too that it assumes a consistent value for month and discount rate is provided in the incoming data.
# MAGIC 
# MAGIC では、spendモデルのカスタムラッパーを定義してみましょう。predict() メソッドは非常にシンプルで、CLVの値を返すだけであることに注意してください。また、入力データから月と割引率の一貫した値が提供されていると仮定していることにも注意してください。
# MAGIC 
# MAGIC Besides modification to the *predict()* method logic, a new definition for *load_context()* is provided.  This method is called when an [mlflow](https://mlflow.org/) model is instantiated.  In it, we will load our lifetimes model artifact:
# MAGIC 
# MAGIC predict()メソッドのロジックの変更に加えて、load_context()の新しい定義が提供されています。このメソッドは、mlflowモデルがインスタンス化されたときに呼び出されます。このメソッドでは、ライフタイムモデルのアーティファクトをロードします。

# COMMAND ----------

import mlflow 
import mlflow.pyfunc

# create wrapper for lifetimes model
class _clvModelWrapper(mlflow.pyfunc.PythonModel):
  
    def __init__(self, spend_model):
      self.spend_model = spend_model
        
    def load_context(self, context):
      # load base model fitter from lifetimes library
      from lifetimes.fitters.base_fitter import BaseFitter
      
      # instantiate lifetimes_model
      self.lifetimes_model = BaseFitter()
      
      # load lifetimes_model from mlflow
      self.lifetimes_model.load_model(context.artifacts['lifetimes_model'])
      
    def predict(self, context, dataframe):
      
      # access input series
      frequency = dataframe.iloc[:,0]
      recency = dataframe.iloc[:,1]
      T = dataframe.iloc[:,2]
      monetary_value = dataframe.iloc[:,3]
      months = int(dataframe.iloc[0,4])
      discount_rate = float(dataframe.iloc[0,5])
      
      # make CLV prediction
      results = pd.DataFrame(
          self.spend_model.customer_lifetime_value(
            self.lifetimes_model, #the model to use to predict the number of future transactions
            frequency,
            recency,
            T,
            monetary_value,
            time=months,
            discount_rate=discount_rate
            ),
          columns=['clv']
          )
      
      return results[['clv']]

# COMMAND ----------

# MAGIC %md Now we save our spend model to mlflow:
# MAGIC 
# MAGIC ここで、spendモデルをmlflowに保存します。

# COMMAND ----------

# add lifetimes to conda environment info
conda_env = mlflow.pyfunc.get_default_conda_env()
# conda_env['dependencies'][1]['pip'] += ['lifetimes==0.10.1'] # version should match version installed at top of this notebook

# save model run to mlflow
with mlflow.start_run(run_name='deployment run') as run:
  
  # identify lifetime model as an artifact associated with the spend model
  artifacts = {'lifetimes_model': lifetimes_model_path}
  
  # log our spend model to mlflow
  mlflow.pyfunc.log_model(
    'model', 
    python_model=_clvModelWrapper(spend_model), 
    conda_env=conda_env,
    artifacts=artifacts
    )

# COMMAND ----------

# MAGIC %md And as before, we create a function from this model:
# MAGIC 
# MAGIC そして、前回と同様に、このモデルから関数を作成します。

# COMMAND ----------

# define the schema of the values returned by the function
result_schema = DoubleType()

# define function based on mlflow recorded model
clv_udf = mlflow.pyfunc.spark_udf(
  spark, 
  'runs:/{0}/model'.format(run.info.run_id), 
  result_type=result_schema
  )

# register the function for use in SQL
_ = spark.udf.register('clv', clv_udf)

# COMMAND ----------

# MAGIC %md Our model is now available for use with the Programmatic SQL API:
# MAGIC 
# MAGIC Programmatic SQL APIで私たちのモデルが使用できるようになりました。

# COMMAND ----------

# create a temp view for SQL demonstration (next cell)
filtered.createOrReplaceTempView('customer_metrics')

# demonstrate function call on Spark DataFrame
display(
  filtered
    .withColumn(
      'clv', 
      clv_udf(filtered.frequency, filtered.recency, filtered.T, filtered.monetary_value, lit(12), lit(0.01))
      )
    .selectExpr(
      'customerid', 
      'clv'
      )
  )

# COMMAND ----------

# MAGIC %md It can also be used with SQL:
# MAGIC 
# MAGIC また、SQLとの併用も可能です。

# COMMAND ----------

# MAGIC %sql -- retreive customer clv
# MAGIC 
# MAGIC SELECT
# MAGIC   customerid,
# MAGIC   clv(
# MAGIC     frequency,
# MAGIC     recency,
# MAGIC     T,
# MAGIC     monetary_value,
# MAGIC     12,
# MAGIC     0.01
# MAGIC     ) as clv
# MAGIC FROM customer_metrics;
