# Databricks notebook source
# MAGIC %md ##Calculating the Probability of Future Customer Engagement
# MAGIC ##将来の顧客エンゲージメントの確率を計算する
# MAGIC 
# MAGIC ※Notebookの原文の下にDeepLでの翻訳結果をそのまま記載しています。
# MAGIC 
# MAGIC In non-subscription retail models, customers come and go with no long-term commitments, making it very difficult to determine whether a customer will return in the future. Determining the probability that a customer will re-engage is critical to the design of effective marketing campaigns. Different messaging and promotions may be required to incentivize customers who have likely dropped out to return to our stores. Engaged customers may be more responsive to marketing that encourages them to expand the breadth and scale of purchases with us. Understanding where our customers land with regard to the probability of future engagement is critical to tailoring our marketing efforts to them.
# MAGIC 
# MAGIC サブスクリプション型ではない小売モデルでは、顧客の出入りが激しく、長期的なコミットメントがないため、顧客が将来的に戻ってくるかどうかを判断することが非常に困難です。そのため、お客様が再び来店されるかどうかを判断することは、効果的なマーケティングキャンペーンを設計する上で非常に重要です。また、一度退店されたお客様が再び来店されるためには、異なるメッセージやプロモーションが必要となる可能性があります。また、当社グループでの購買の幅や規模を拡大するようなマーケティングを行えば、顧客はより積極的に反応する可能性があります。このように、お客様が将来的にどのような行動を取る可能性があるのかを把握することは、お客様に合わせてマーケティング活動を行う上で非常に重要です。
# MAGIC 
# MAGIC The *Buy 'til You Die* (BTYD) models popularized by Peter Fader and others leverage two basic customer metrics, *i.e.* the recency of a customer's last engagement and the frequency of repeat transactions over a customer's lifetime, to derive a probability of future re-engagement. This is done by fitting customer history to curves describing the distribution of purchase frequencies and engagement drop-off following a prior purchase. The math behind these models is fairly complex but thankfully it's been encapsulated in the [lifetimes](https://pypi.org/project/Lifetimes/) library, making it much easier for traditional enterprises to employ. The purpose of this notebook is to examine how these models may be applied to customer transaction history and how they may be deployed for integration in marketing processes.
# MAGIC 
# MAGIC ピーター・フェーダーなどによって普及した BTYD（Buy 'til You Die）モデルは、2 つの基本的な顧客指標、つまり、顧客の最後のエンゲージメントの再帰性と、顧客の生涯におけるリピート取引の頻度を活用して、将来の再エンゲージメント確率を導き出すものである。これは、購入頻度の分布と、前回の購入後のエンゲージメントの低下を記述する曲線に、顧客の履歴を当てはめることで行われる。これらのモデルの背後にある数学はかなり複雑ですが、幸いにもライフタイムライブラリにカプセル化されているため、従来の企業でも簡単に採用することができます。このノートブックの目的は、これらのモデルをどのように顧客の取引履歴に適用し、マーケティングプロセスに統合するためにどのように展開するかを検討することです。

# COMMAND ----------

# MAGIC %md ###Step 1: Setup the Environment
# MAGIC ステップ1：環境の構築
# MAGIC 
# MAGIC To run this notebook, you need to attach to a **Databricks ML Runtime** cluster leveraging Databricks version 6.5+. This version of the Databricks runtime will provide access to many of the pre-configured libraries used here.  Still, there are additional Python libraries which you will need to install and attach to your cluster.  These are:</p>
# MAGIC 
# MAGIC このノートブックを実行するには、Databricksバージョン6.5+を活用したDatabricks ML Runtimeクラスタに接続する必要があります。このバージョンのDatabricksランタイムは、ここで使用する設定済みのライブラリの多くにアクセスすることができます。しかし、追加のPythonライブラリをインストールし、クラスタにアタッチする必要があります。これらは
# MAGIC * xlrd
# MAGIC * lifetimes==0.10.1
# MAGIC * nbconvert
# MAGIC 
# MAGIC To install these libraries in your Databricks workspace, please follow [these steps](https://docs.databricks.com/libraries.html#workspace-libraries) using the PyPI library source in combination with the bullet-pointed library names in the provided list.  Once installed, please be sure to [attach](https://docs.databricks.com/libraries.html#install-a-library-on-a-cluster) these libraries to the cluster with which you are running this notebook.
# MAGIC 
# MAGIC これらのライブラリをDatabricksのワークスペースにインストールするには、PyPIのライブラリソースと提供されたリストの箇条書きのライブラリ名を組み合わせて、次の手順に従ってください。インストールしたら、これらのライブラリをこのノートブックを実行しているクラスタにアタッチしてください。
# MAGIC 
# MAGIC ※原追記
# MAGIC xlrdはxlsファイルにしか対応しなくなったそうで、openpyxlを追加インストールしています。

# COMMAND ----------

# MAGIC %md With the libraries installed, let's load a sample dataset with which we can examine the BTYD models. The dataset we will use is the [Online Retail Data Set](http://archive.ics.uci.edu/ml/datasets/Online+Retail) available from the UCI Machine Learning Repository.  This dataset is made available as a Microsoft Excel workbook (XLSX).  Having downloaded this XLSX file to our local system, we can load it into our Databricks environment by following the steps provided [here](https://docs.databricks.com/data/tables.html#create-table-ui). Please note when performing the file import, you don't need to select the *Create Table with UI* or the *Create Table in Notebook* options to complete the import process. Also, the name of the XLSX file will be modified upon import as it includes an unsupported space character.  As a result, we will need to programmatically locate the new name for the file assigned by the import process.
# MAGIC 
# MAGIC Assuming we've uploaded the XLSX to the */FileStore/tables/online_retail/*, we can access it as follows:
# MAGIC 
# MAGIC ライブラリがインストールされた状態で、BTYDモデルを検証するためのサンプルデータセットをロードしてみましょう。今回使用するデータセットは、UCI Machine Learning Repositoryから入手可能なOnline Retail Data Setです。このデータセットはMicrosoft Excelワークブック(XLSX)として提供されています。このXLSXファイルをローカルシステムにダウンロードした後、ここに記載されている手順に従って、Databricks環境にロードすることができます。ファイルのインポートを行う際、「Create Table with UI」または「Create Table in Notebook」オプションを選択しなくても、インポートプロセスは完了することに注意してください。また、XLSXファイルの名前にサポートされていないスペース文字が含まれているため、インポート時に変更されます。そのため、インポート処理で割り当てられたファイルの新しい名前をプログラムによって特定する必要があります。
# MAGIC 
# MAGIC XLSXを/FileStore/tables/online_retail/にアップロードしたと仮定すると、次のようにアクセスすることができます。

# COMMAND ----------

import pandas as pd
import numpy as np

# identify name of xlsx file (which will change when uploaded)
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
# the xlrd library must be installed for this step to work 
orders_pd = pd.read_excel(
  xlsx_filename, 
  sheet_name='Online Retail',
  header=0, # first row is header
  dtype=orders_schema
  )

# display first few rows from the dataset
orders_pd.head(10)

# COMMAND ----------

# MAGIC %md The data in the workbook are organized as a range in the Online Retail spreadsheet.  Each record represents a line item in a sales transaction. The fields included in the dataset are:
# MAGIC 
# MAGIC ワークブックのデータは、Online Retailスプレッドシートの範囲として整理されています。各レコードは、販売トランザクションの行項目を表します。データセットに含まれるフィールドは次のとおりです。
# MAGIC 
# MAGIC | Field | Description |
# MAGIC |-------------:|-----:|
# MAGIC |InvoiceNo|A 6-digit integral number uniquely assigned to each transaction<br/>各取引に一意的に割り当てられた6桁の整数値|
# MAGIC |StockCode|A 5-digit integral number uniquely assigned to each distinct product<br/>各商品に一意的に割り当てられた5桁の整数値|
# MAGIC |Description|The product (item) name<br/>商品（アイテム）名|
# MAGIC |Quantity|The quantities of each product (item) per transaction<br/>各商品（アイテム）の1取引あたりの数量|
# MAGIC |InvoiceDate|The invoice date and a time in mm/dd/yy hh:mm format<br/>請求日および時刻（mm/dd/yy hh:mm形式）|
# MAGIC |UnitPrice|The per-unit product price in pound sterling (£)<br/>製品1個あたりの価格（ポンド建て）|
# MAGIC |CustomerID| A 5-digit integral number uniquely assigned to each customer<br/>各顧客に一意に割り当てられた5桁のインテグラルナンバー|
# MAGIC |Country|The name of the country where each customer resides<br/>お客様が居住する国名|
# MAGIC 
# MAGIC Of these fields, the ones of particular interest for our work are InvoiceNo which identifies the transaction, InvoiceDate which identifies the date of that transaction, and CustomerID which uniquely identifies the customer across multiple transactions. (In a separate notebook, we will examine the monetary value of the transactions through the UnitPrice and Quantity fields.)
# MAGIC 
# MAGIC これらのフィールドのうち、今回の作業で特に注目するのは、取引を特定するInvoiceNo、その取引日を特定するInvoiceDate、複数の取引にまたがって顧客を一意に特定するCustomerIDの3つである。(別のノートブックでは、UnitPriceとQuantityフィールドを通じて取引の金銭的価値を調べる予定です)。

# COMMAND ----------

# MAGIC %md ###Step 2: Explore the Dataset
# MAGIC ###ステップ2：データセットの探索
# MAGIC 
# MAGIC To enable the exploration of the data using SQL statements, let's flip the pandas DataFrame into a Spark DataFrame and persist it as a temporary view:
# MAGIC 
# MAGIC SQL文を使ってデータを探索できるようにするために、pandas DataFrameをSpark DataFrameに反転し、一時的なビューとして永続化させましょう。

# COMMAND ----------

# convert pandas DF to Spark DF
orders = spark.createDataFrame(orders_pd)

# present Spark DF as queriable view
orders.createOrReplaceTempView('orders') 

# COMMAND ----------

# MAGIC %md Examining the transaction activity in our dataset, we can see the first transaction occurs December 1, 2010 and the last is on December 9, 2011 making this a dataset that's a little more than 1 year in duration. The daily transaction count shows there is quite a bit of volatility in daily activity for this online retailer:
# MAGIC 
# MAGIC このデータセットの取引活動を調べると、最初の取引は2010年12月1日、最後の取引は2011年12月9日であり、1年余りの期間にわたるデータセットであることがわかります。1日の取引回数を見ると、このオンラインショップの1日のアクティビティにはかなりの変動があることがわかります。

# COMMAND ----------

# MAGIC %sql -- unique transactions by date
# MAGIC 
# MAGIC SELECT 
# MAGIC   TO_DATE(InvoiceDate) as InvoiceDate,
# MAGIC   COUNT(DISTINCT InvoiceNo) as Transactions
# MAGIC FROM orders
# MAGIC GROUP BY TO_DATE(InvoiceDate)
# MAGIC ORDER BY InvoiceDate;

# COMMAND ----------

# MAGIC %md We can smooth this out a bit by summarizing activity by month. It's important to keep in mind that December 2011 only consists of 9 days so the sales decline graphed for the last month should most likely be ignored:
# MAGIC 
# MAGIC NOTE We will hide the SQL behind each of the following result sets for ease of viewing.  To view this code, simply click the **Show code** item above each of the following charts.
# MAGIC 
# MAGIC 月別のアクティビティをまとめると、少し滑らかになります。2011年12月は9日間しかないため、先月にグラフ化された売上高の減少は無視することができます。
# MAGIC 
# MAGIC 注：以下の結果セットの背後にあるSQLは、見やすくするために非表示にしています。このコードを表示するには、以下の各チャートの上にある Show code アイテムをクリックするだけです。

# COMMAND ----------

# MAGIC %sql -- unique transactions by month
# MAGIC 
# MAGIC SELECT 
# MAGIC   TRUNC(InvoiceDate, 'month') as InvoiceMonth,
# MAGIC   COUNT(DISTINCT InvoiceNo) as Transactions
# MAGIC FROM orders
# MAGIC GROUP BY TRUNC(InvoiceDate, 'month') 
# MAGIC ORDER BY InvoiceMonth;

# COMMAND ----------

# MAGIC %md For the little more than 1-year period for which we have data, we see over four-thousand unique customers.  These customers generated about twenty-two thousand unique transactions:
# MAGIC 
# MAGIC 1年強のデータのうち、ユニークカスタマー数は4,000人を超えています。これらの顧客は、約22,000件のユニークなトランザクションを生成しました。

# COMMAND ----------

# MAGIC %sql -- unique customers and transactions
# MAGIC 
# MAGIC SELECT
# MAGIC  COUNT(DISTINCT CustomerID) as Customers,
# MAGIC  COUNT(DISTINCT InvoiceNo) as Transactions
# MAGIC FROM orders
# MAGIC WHERE CustomerID IS NOT NULL;

# COMMAND ----------

# MAGIC %md A little quick math may lead us to estimate that, on average, each customer is responsible for about 5 transactions, but this would not provide an accurate representation of customer activity.
# MAGIC 
# MAGIC Instead, if we count the unique transactions by customer and then examine the frequency of these values, we see that many of the customers have engaged in a single transaction. The distribution of the count of repeat purchases declines from there in a manner that we may describe as negative binomial distribution (which is the basis of the NBD acronym included in the name of most BTYD models):
# MAGIC 
# MAGIC 少し計算すると、1人の顧客が平均して約5件の取引を担当していると推定されるが、これでは顧客の活動を正確に表すことはできない。
# MAGIC 
# MAGIC その代わり、顧客ごとのユニークな取引をカウントし、その値の頻度を調べると、多くの顧客が1回の取引に従事していることがわかる。リピート購入回数の分布は、そこから負の二項分布（ほとんどのBTYDモデルの名称に含まれるNBDの頭文字の基になっている）と表現できるような形で減少していくのです。

# COMMAND ----------

# MAGIC %sql -- the distribution of per-customer transaction counts 
# MAGIC 
# MAGIC SELECT
# MAGIC   x.Transactions,
# MAGIC   COUNT(x.*) as Occurrences
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC     CustomerID,
# MAGIC     COUNT(DISTINCT InvoiceNo) as Transactions 
# MAGIC   FROM orders
# MAGIC   WHERE CustomerID IS NOT NULL
# MAGIC   GROUP BY CustomerID
# MAGIC   ) x
# MAGIC GROUP BY 
# MAGIC   x.Transactions
# MAGIC ORDER BY
# MAGIC   x.Transactions;

# COMMAND ----------

# MAGIC %md If we alter our last analysis to group a customer's transactions that occur on the same date into a single transaction - a pattern that aligns with metrics we will calculate later - we see that a few more customers are identified as non-repeat customers but the overall pattern remains the same:
# MAGIC 
# MAGIC 前回の分析を変更して、同じ日に発生した顧客の取引を1つの取引にグループ化すると（このパターンは、後で計算する指標と一致します）、非リピート顧客として識別される顧客が少し増えることがわかりますが、全体のパターンは同じままです。

# COMMAND ----------

# MAGIC %sql -- the distribution of per-customer transaction counts
# MAGIC      -- with consideration of same-day transactions as a single transaction 
# MAGIC 
# MAGIC SELECT
# MAGIC   x.Transactions,
# MAGIC   COUNT(x.*) as Occurances
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC     CustomerID,
# MAGIC     COUNT(DISTINCT TO_DATE(InvoiceDate)) as Transactions
# MAGIC   FROM orders
# MAGIC   WHERE CustomerID IS NOT NULL
# MAGIC   GROUP BY CustomerID
# MAGIC   ) x
# MAGIC GROUP BY 
# MAGIC   x.Transactions
# MAGIC ORDER BY
# MAGIC   x.Transactions;

# COMMAND ----------

# MAGIC %md Focusing on customers with repeat purchases, we can examine the distribution of the days between purchase events. What's important to note here is that most customers return to the site within 2 to 3 months of a prior purchase.  Longer gaps do occur but significantly fewer customers have longer gaps between returns.  This is important to understand in the context of our BYTD models in that the time since we last saw a customer is a critical factor to determining whether they will ever come back with the probability of return dropping as more and more time passes since a customer's last purchase event:
# MAGIC 
# MAGIC リピート購入する顧客に着目し、購入イベント間の日数分布を調べることができる。ここで注目すべきは、ほとんどのお客様が、前回の購入から2〜3ヶ月以内に再来店していることです。これより長い間隔を空けることもありますが、それ以上の間隔を空ける顧客はかなり少なくなっています。このことは、BYTD モデルの文脈で理解することが重要です。つまり、顧客に最後に会ってからの時間が、顧客が再び訪れるかどうかを決定する重要な要因であり、顧客の最後の購入イベントから時間が経過するほど、再び訪れる確率が低下するのです。

# COMMAND ----------

# MAGIC %sql -- distribution of per-customer average number of days between purchase events
# MAGIC 
# MAGIC WITH CustomerPurchaseDates
# MAGIC   AS (
# MAGIC     SELECT DISTINCT
# MAGIC       CustomerID,
# MAGIC       TO_DATE(InvoiceDate) as InvoiceDate
# MAGIC     FROM orders 
# MAGIC     WHERE CustomerId IS NOT NULL
# MAGIC     )
# MAGIC SELECT -- Per-Customer Average Days Between Purchase Events
# MAGIC   AVG(
# MAGIC     DATEDIFF(a.NextInvoiceDate, a.InvoiceDate)
# MAGIC     ) as AvgDaysBetween
# MAGIC FROM ( -- Purchase Event and Next Purchase Event by Customer
# MAGIC   SELECT 
# MAGIC     x.CustomerID,
# MAGIC     x.InvoiceDate,
# MAGIC     MIN(y.InvoiceDate) as NextInvoiceDate
# MAGIC   FROM CustomerPurchaseDates x
# MAGIC   INNER JOIN CustomerPurchaseDates y
# MAGIC     ON x.CustomerID=y.CustomerID AND x.InvoiceDate < y.InvoiceDate
# MAGIC   GROUP BY 
# MAGIC     x.CustomerID,
# MAGIC     x.InvoiceDate
# MAGIC     ) a
# MAGIC GROUP BY CustomerID

# COMMAND ----------

# MAGIC %md ###Step 3: Calculate Customer Metrics
# MAGIC ###ステップ3：顧客指標を算出する
# MAGIC 
# MAGIC The dataset with which we are working consists of raw transactional history.  To apply the BTYD models, we need to derive several per-customer metrics:</p>
# MAGIC 
# MAGIC 今回扱うデータセットは、生の取引履歴から構成されている。BTYD モデルを適用するために、いくつかの顧客ごとの指標を導き出す必要があります。
# MAGIC 
# MAGIC * **Frequency** - the number of dates on which a customer made a purchase subsequent to the date of the customer's first purchase
# MAGIC * **Age (T)** - the number of time units, *e.g.* days, since the date of a customer's first purchase to the current date (or last date in the dataset)　
# MAGIC * **Recency** - the age of the customer (as previously defined) at the time of their last purchase
# MAGIC 
# MAGIC * **Frequency** - 顧客が最初に購入した日以降に購入を行った日付の数
# MAGIC * **Age (T)** - 顧客の最初の購入日から現在の日付（またはデータセットの最後の日付）までの、日数などの時間単位の数
# MAGIC * **Recency** - 最後に購入した時の顧客の年齢（前に定義したとおり）。
# MAGIC 
# MAGIC It's important to note that when calculating metrics such as customer age that we need to consider when our dataset terminates.  Calculating these metrics relative to today's date can lead to erroneous results.  Given this, we will identify the last date in the dataset and define that as *today's date* for all calculations.
# MAGIC 
# MAGIC 顧客年齢などの指標を計算する際に、データセットがいつ終了するかを考慮する必要があることに注意することが重要である。今日の日付を基準にしてこれらのメトリクスを計算すると、誤った結果になる可能性があります。そこで、データセットの最後の日付を特定し、それをすべての計算のための今日の日付として定義します。
# MAGIC 
# MAGIC To get started with these calculations, let's take a look at how they are performed using the built-in functionality of the lifetimes library:
# MAGIC 
# MAGIC これらの計算を始めるために、lifetimesライブラリの組み込み機能を使って、どのように計算を行うか見てみましょう。

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
    freq='D'
    )
  )

# display first few rows
metrics_pd.head(10)

# COMMAND ----------

# MAGIC %md The lifetimes library, like many Python libraries, is single-threaded.  Using this library to derive customer metrics on larger transactional datasets may overwhelm your system or simply take too long to complete. For this reason, let's examine how these metrics can be calculated using the distributed capabilities of Apache Spark.
# MAGIC 
# MAGIC lifetimesライブラリは、多くのPythonライブラリと同様、シングルスレッドです。このライブラリを使用して、より大きなトランザクションデータセットから顧客メトリクスを導出すると、システムに負荷がかかったり、単純に時間がかかりすぎたりする可能性があります。このため、Apache Sparkの分散機能を使用して、これらのメトリクスを計算する方法を検討します。
# MAGIC 
# MAGIC As SQL is frequency employed for complex data manipulation, we'll start with a Spark SQL statement.  In this statement, we first assemble each customer's order history consisting of the customer's ID, the date of their first purchase (first_at), the date on which a purchase was observed (transaction_at) and the current date (using the last date in the dataset for this value).  From this history, we can count the number of repeat transaction dates (frequency), the days between the last and first transaction dates (recency), and the days between the current date and first transaction (T) on a per-customer basis:
# MAGIC 
# MAGIC 複雑なデータ操作にはSQLが頻繁に使用されるので、まずはSpark SQLステートメントを使用します。このステートメントでは、まず、顧客のID、最初の購入日（first_at）、購入が確認された日（transaction_at）、現在の日付（この値にはデータセットの最後の日付を使用）から成る各顧客の注文履歴を組み立てます。この履歴から、顧客ごとに、取引日を繰り返す回数（frequency）、最後の取引日から最初の取引日までの日数（recency）、現在の日付から最初の取引日までの日数（T）をカウントすることができます。

# COMMAND ----------

# sql statement to derive summary customer stats
sql = '''
  SELECT
    a.customerid as CustomerID,
    CAST(COUNT(DISTINCT a.transaction_at) - 1 as float) as frequency,
    CAST(DATEDIFF(MAX(a.transaction_at), a.first_at) as float) as recency,
    CAST(DATEDIFF(a.current_dt, a.first_at) as float) as T
  FROM ( -- customer order history
    SELECT DISTINCT
      x.customerid,
      z.first_at,
      TO_DATE(x.invoicedate) as transaction_at,
      y.current_dt
    FROM orders x
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

# MAGIC %md Of course, Spark SQL does not require the DataFrame to be accessed exclusively using a SQL statement.  We may derive this same result using the Programmatic SQL API which may align better with some Data Scientist's preferences.  The code in the next cell is purposely assembled to mirror the structure in the previous SQL statement for the purposes of comparison:
# MAGIC 
# MAGIC もちろん、Spark SQLでは、DataFrameにSQL文のみでアクセスする必要はありません。データサイエンティストの好みに合わせて、Programmatic SQL APIを使って同じ結果を導き出すこともできます。次のセルのコードは、比較のために、前のSQL文の構造をミラーリングするように意図的に組み立てられています。

# COMMAND ----------

from pyspark.sql.functions import to_date, datediff, max, min, countDistinct, count, sum, when
from pyspark.sql.types import *

# valid customer orders
x = orders.where(orders.CustomerID.isNotNull())

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
      to_date(x.InvoiceDate).alias('transaction_at'), 
      y.current_dt
      )
     .distinct()
    )

# calculate relevant metrics by customer
metrics_api = (a
           .groupBy(a.customerid, a.current_dt, a.first_at)
           .agg(
             (countDistinct(a.transaction_at)-1).cast(FloatType()).alias('frequency'),
             datediff(max(a.transaction_at), a.first_at).cast(FloatType()).alias('recency'),
             datediff(a.current_dt, a.first_at).cast(FloatType()).alias('T')
             )
           .select('customerid','frequency','recency','T')
           .orderBy('customerid')
          )

display(metrics_api)

# COMMAND ----------

# MAGIC %md Let's take a moment to compare the data in these different metrics datasets, just to confirm the results are identical.  Instead of doing this record by record, let's calculate summary statistics across each dataset to verify their consistency:
# MAGIC 
# MAGIC これらの異なる測定基準データセットのデータを比較し、結果が同じであることを確認してみましょう。レコードごとに行うのではなく、各データセットの要約統計量を計算して、その一貫性を検証してみましょう。
# MAGIC 
# MAGIC NOTE You may notice means and standard deviations vary slightly in the hundred-thousandths and millionths decimal places.  This is a result of slight differences in data types between the pandas and Spark DataFrames but do not affect our results in a meaningful way. 
# MAGIC 
# MAGIC 注：平均値と標準偏差は、小数点以下10万分の1や100万分の1でわずかに異なることに気づくかもしれません。これはpandasとSpark DataFrameのデータ型のわずかな違いによるものですが、結果に大きな影響を与えるものではありません。

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

# MAGIC %md The metrics we've calculated represent summaries of a time series of data.  To support model validation and avoid overfitting, a common pattern with time series data is to train models on an earlier portion of the time series (known as the *calibration* period) and validate against a later portion of the time series (known as the *holdout* period). In the lifetimes library, the derivation of per customer metrics using calibration and holdout periods is done through a simple method call.  Because our dataset consists of a limited range for data, we will instruct this library method to use the last 90-days of data as the holdout period.  A simple parameter called a widget on the Databricks platform has been implemented to make the configuration of this setting easily changeable:
# MAGIC 
# MAGIC 今回計算したメトリクスは、時系列データのサマリーを表しています。モデルの検証をサポートし、オーバーフィッティングを避けるために、時系列データの一般的なパターンは、時系列の早い部分（キャリブレーション期間として知られている）でモデルを訓練し、時系列の遅い部分（ホールドアウト期間として知られている）に対して検証を行うことです。lifetimesライブラリでは、キャリブレーション期間とホールドアウト期間を使用した顧客ごとのメトリックの導出は、シンプルなメソッドコールで行われます。今回のデータセットは限られた範囲のデータで構成されているため、ホールドアウト期間として直近の90日間のデータを使用するよう、このライブラリのメソッドに指示します。Databricksプラットフォームでは、この設定を簡単に変更できるように、ウィジェットと呼ばれる簡単なパラメータが実装されています。
# MAGIC 
# MAGIC NOTE To change the number of days in the holdout period, look for the textbox widget by scrolling to the top of your Databricks notebook after running this next cell
# MAGIC 
# MAGIC 注：ホールドアウト期間の日数を変更するには、次のセルを実行した後、Databricksノートブックの一番上にスクロールしてテキストボックスウィジェットを探します。

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
    freq='D'    
    )
  )

# display first few rows
metrics_cal_pd.head(10)

# COMMAND ----------

# MAGIC %md As before, we may leverage Spark SQL to derive this same information.  Again, we'll examine this through both a SQL statement and the programmatic SQL API.
# MAGIC 
# MAGIC 前回と同様に、Spark SQLを活用して、同じ情報を導き出すことができます。ここでも、SQL文とプログラムによるSQL APIの両方を通して検証していきます。
# MAGIC 
# MAGIC To understand the SQL statement, first recognize that it's divided into two main parts.  In the first, we calculate the core metrics, *i.e.* recency, frequency and age (T), per customer for the calibration period, much like we did in the previous query example. In the second part of the query, we calculate the number of purchase dates in the holdout customer for each customer.  This value (frequency_holdout) represents the incremental value to be added to the frequency for the calibration period (frequency_cal) when we examine a customer's entire transaction history across both calibration and holdout periods.
# MAGIC 
# MAGIC SQL文を理解するために、まず、SQL文が大きく2つのパートに分かれていることを認識します。最初の部分では、前のクエリ例で行ったように、校正期間の顧客ごとのコアメトリクス、すなわち recency、frequency、age (T)を計算します。クエリーの2つ目の部分では、各顧客について、ホールドアウトの顧客の購入日の回数を計算します。この値（frequency_holdout）は、校正期間とホールドアウト期間の両方にわたって顧客の取引履歴全体を調べたときに、校正期間の頻度（frequency_cal）に追加される増分を表します。
# MAGIC 
# MAGIC To simplify our logic, a common table expression (CTE) named CustomerHistory is defined at the top of the query.  This query extracts the relevant dates that make up a customer's transaction history and closely mirrors the logic at the center of the last SQL statement we examined.  The only difference is that we include the number of days in the holdout period (duration_holdout):
# MAGIC 
# MAGIC ロジックを単純化するために、CustomerHistory という名前の共通テーブル式 (CTE) がクエリの先頭で定義されています。このクエリは、顧客の取引履歴を構成する関連する日付を抽出し、前回調べたSQL文の中心にあるロジックとほぼ同じです。唯一の違いは、保留期間の日数(duration_holdout)を含めることです。
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
      SELECT DISTINCT
        x.customerid,
        z.first_at,
        TO_DATE(x.invoicedate) as transaction_at,
        y.current_dt
      FROM orders x
      CROSS JOIN (SELECT MAX(TO_DATE(invoicedate)) as current_dt FROM orders) y                                -- current date (according to dataset)
      INNER JOIN (SELECT customerid, MIN(TO_DATE(invoicedate)) as first_at FROM orders GROUP BY customerid) z  -- first order per customer
        ON x.customerid=z.customerid
      WHERE x.customerid IS NOT NULL
    ) m
  )
SELECT
    a.customerid as CustomerID,
    a.frequency as frequency_cal,
    a.recency as recency_cal,
    a.T as T_cal,
    COALESCE(b.frequency_holdout, 0.0) as frequency_holdout,
    a.duration_holdout
FROM ( -- CALIBRATION PERIOD CALCULATIONS
    SELECT
        p.customerid,
        CAST(p.duration_holdout as float) as duration_holdout,
        CAST(DATEDIFF(MAX(p.transaction_at), p.first_at) as float) as recency,
        CAST(COUNT(DISTINCT p.transaction_at) - 1 as float) as frequency,
        CAST(DATEDIFF(DATE_SUB(p.current_dt, p.duration_holdout), p.first_at) as float) as T
    FROM CustomerHistory p
    WHERE p.transaction_at < DATE_SUB(p.current_dt, p.duration_holdout)  -- LIMIT THIS QUERY TO DATA IN THE CALIBRATION PERIOD
    GROUP BY p.customerid, p.first_at, p.current_dt, p.duration_holdout
  ) a
LEFT OUTER JOIN ( -- HOLDOUT PERIOD CALCULATIONS
  SELECT
    p.customerid,
    CAST(COUNT(DISTINCT p.transaction_at) as float) as frequency_holdout
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

# MAGIC %md And here is the equivalent Programmatic SQL API logic:
# MAGIC 
# MAGIC そして、これに相当するProgrammatic SQL APIのロジックは以下の通りです。

# COMMAND ----------

from pyspark.sql.functions import avg, date_sub, coalesce, lit, expr

# valid customer orders
x = orders.where(orders.CustomerID.isNotNull())

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
      to_date(x.InvoiceDate).alias('transaction_at'), 
      y.current_dt, 
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
         datediff( expr('date_sub(current_dt, duration_holdout)'), p.first_at).cast(FloatType()).alias('T_cal')
       )
    )

b = (p
      .where((p.transaction_at >= expr('date_sub(current_dt, duration_holdout)')) & (p.transaction_at <= p.current_dt) )
      .groupBy(p.customerid)
      .agg(
        countDistinct(p.transaction_at).cast(FloatType()).alias('frequency_holdout')
        )
   )

metrics_cal_api = (a
                 .join(b, a.customerid==b.customerid, how='left')
                 .select(
                   a.customerid.alias('CustomerID'),
                   a.frequency_cal,
                   a.recency_cal,
                   a.T_cal,
                   coalesce(b.frequency_holdout, lit(0.0)).alias('frequency_holdout'),
                   a.duration_holdout
                   )
                 .orderBy('CustomerID')
              )

display(metrics_cal_api)

# COMMAND ----------

# MAGIC %md Using summary stats, we can again verify these different units of logic are returning the same results:
# MAGIC 
# MAGIC サマリー統計を使って、これらの異なるロジック単位が同じ結果を返していることを再度確認することができます。

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

# MAGIC %md Our data prep is nearly done.  The last thing we need to do is exclude customers for which we have no repeat purchases, *i.e.* frequency or frequency_cal is 0. The Pareto/NBD and BG/NBD models we will use focus exclusively on performing calculations on customers with repeat transactions.  A modified BG/NBD model, *i.e.* MBG/NBD, which allows for customers with no repeat transactions is supported by the lifetimes library.  However, to stick with the two most popular of the BYTD models in use today, we will limit our data to align with their requirements:
# MAGIC 
# MAGIC データの下準備はほぼ完了しました。最後に必要なことは、リピート購入がない顧客、つまりfrequencyまたはfrequency_calが0の顧客を除外することです。私たちが使用するPareto/NBDおよびBG/NBDモデルは、リピート取引のある顧客に対してのみ計算を実行することに焦点を当てています。BG/NBDモデルの修正版、すなわちMBG/NBDは、繰り返し取引のない顧客を許容するもので、lifetimesライブラリでサポートされています。しかし、現在使用されているBYTDモデルのうち最も一般的な2つのモデルにこだわるため、その要件に沿うようにデータを制限することにします。
# MAGIC 
# MAGIC NOTE We are showing how both the pandas and Spark DataFrames are filtered simply to be consistent with side-by-side comparisons earlier in this section of the notebook.  In a real-world implementation, you would simply choose to work with pandas or Spark DataFrames for data preparation.
# MAGIC 
# MAGIC 注：pandasとSpark DataFrameの両方がどのようにフィルタリングされているかを示しているのは、このノートブックのこのセクションの前のサイドバイサイドの比較と整合性を持たせるためです。実際の実装では、データ準備のためにpandasまたはSpark DataFramesのどちらかを選択するだけです。

# COMMAND ----------

# remove customers with no repeats (complete dataset)
filtered_pd = metrics_pd[metrics_pd['frequency'] > 0]
filtered = metrics_api.where(metrics_api.frequency > 0)

## remove customers with no repeats in calibration period
filtered_cal_pd = metrics_cal_pd[metrics_cal_pd['frequency_cal'] > 0]
filtered_cal = metrics_cal_api.where(metrics_cal_api.frequency_cal > 0)

# COMMAND ----------

# MAGIC %md ###Step 4: Train the Model
# MAGIC 
# MAGIC ###ステップ4：モデルの訓練
# MAGIC 
# MAGIC To ease into the training of a model, let's start with a simple exercise using a [Pareto/NBD model](https://www.jstor.org/stable/2631608), the original BTYD model. We'll use the calibration-holdout dataset constructed in the last section of this notebook, fitting the model to the calibration data and later evaluating it using the holdout data:
# MAGIC 
# MAGIC モデルの学習を簡単にするために、まずはPareto/NBDモデル、つまりオリジナルのBTYDモデルを使った簡単な練習をしましょう。このノートブックの最後のセクションで作成したキャリブレーションとホールドアウトのデータセットを使い、キャリブレーションデータにモデルを当てはめ、後でホールドアウトデータを使って評価します。

# COMMAND ----------

from lifetimes.fitters.pareto_nbd_fitter import ParetoNBDFitter
from lifetimes.fitters.beta_geo_fitter import BetaGeoFitter

# load spark dataframe to pandas dataframe
input_pd = filtered_cal.toPandas()

# fit a model
model = ParetoNBDFitter(penalizer_coef=0.0)
model.fit( input_pd['frequency_cal'], input_pd['recency_cal'], input_pd['T_cal'])

# COMMAND ----------

# MAGIC %md With our model now fit, let's make some predictions for the holdout period.  We'll grab the actuals for that same period to enable comparison in a subsequent step:
# MAGIC 
# MAGIC モデルの適合が完了したので、保留期間の予測をしてみましょう。次のステップで比較できるように、同じ期間の実績値を取得します。

# COMMAND ----------

# get predicted frequency during holdout period
frequency_holdout_predicted = model.predict( input_pd['duration_holdout'], input_pd['frequency_cal'], input_pd['recency_cal'], input_pd['T_cal'])

# get actual frequency during holdout period
frequency_holdout_actual = input_pd['frequency_holdout']

# COMMAND ----------

# MAGIC %md With actual and predicted values in hand, we can calculate some standard evaluation metrics.  Let's wrap those calculations in a function call to make evaluation easier in future steps:
# MAGIC 
# MAGIC 実測値と予測値が手元にあれば、いくつかの標準的な評価指標を計算することができます。これらの計算を関数呼び出しでラップして、今後のステップで評価を簡単に行えるようにしましょう。

# COMMAND ----------

import numpy as np

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
    val = np.sum(np.abs(actuals-predicted))/actuals.shape[0]
  
  else:
    val = None
  
  return val

# score the model
print('MSE: {0}'.format(score_model(frequency_holdout_actual, frequency_holdout_predicted, 'mse')))

# COMMAND ----------

# MAGIC %md While the internals of the Pareto/NBD model may be quite complex.  In a nutshell, the model calculates a double integral of two curves, one which describes the frequency of customer purchases within a population and another which describes customer survivorship following a prior purchase event. All of the calculation logic is thankfully hidden behind a simple method call.
# MAGIC 
# MAGIC パレート/NBDモデルの内部は非常に複雑である。1つは母集団内の顧客の購入頻度を表す曲線で、もう1つは以前の購入イベントの後の顧客の生存率を表す曲線です。計算ロジックはすべて、単純なメソッド呼び出しの後ろに隠されているのがありがたい。
# MAGIC 
# MAGIC As simple as training a model may be, we have two models that we could use here: the Pareto/NBD model and the BG/NBD model.  The [BG/NBD model](http://brucehardie.com/papers/018/fader_et_al_mksc_05.pdf) simplifies the math involved in calculating customer lifetime and is the model that popularized the BTYD approach.  Both models work off the same customer features and employ the same constraints.  (The primary difference between the two models is that the BG/NBD model maps the survivorship curve to a beta-geometric distribution instead of a Pareto distribution.) To achieve the best fit possible, it is worthwhile to compare the results of both models with our dataset.
# MAGIC 
# MAGIC モデルのトレーニングは簡単ですが、ここではPareto/NBDモデルとBG/NBDモデルの2つのモデルを使用します。BG/NBD モデルは、顧客ライフタイムの計算に関わる計算を単純化し、BTYD アプローチを普及させたモデルです。どちらのモデルも、同じ顧客の特徴を利用しており、同じ制約を採用しています。(2つのモデルの主な違いは、BG/NBDモデルが生存曲線をパレート分布ではなくベータ幾何分布にマッピングしていることです)。可能な限り最適な適合を得るために、両モデルの結果を我々のデータセットで比較することは価値があります。
# MAGIC 
# MAGIC Each model leverages an L2-norm regularization parameter which we've arbitrarily set to 0 in the previous training cycle.  In addition to exploring which model works best, we should consider which value (between 0 and 1) works best for this parameter.  This gives us a pretty broad search space to explore with some hyperparameter tuning.
# MAGIC 
# MAGIC 各モデルはL2-norm正則化パラメータを利用しており、これは前回の学習サイクルで任意に0に設定したものである。どのモデルが最適かを検討するだけでなく、このパラメータにどの値（0と1の間）が最適かを検討する必要があります。これにより、ハイパーパラメータを調整することで、かなり広い探索空間が得られます。
# MAGIC 
# MAGIC To assist us with this, we will make use of [hyperopt](http://hyperopt.github.io/hyperopt/).  Hyperopt allows us to parallelize the training and evaluation of models against a hyperparameter search space.  This can be done leveraging the multiprocessor resources of a single machine or across the broader resources provided by a Spark cluster.  With each model iteration, a loss function is calculated.  Using various optimization algorithms, hyperopt navigates the search space to locate the best available combination of parameter settings to minimize the value returned by the loss function.
# MAGIC 
# MAGIC これを支援するために、hyperoptを使用することにします。hyperoptは、ハイパーパラメータの探索空間に対して、モデルの学習と評価を並列化することができます。これは、1台のマシンのマルチプロセッサ・リソースを活用することも、Sparkクラスタが提供する広範なリソースにまたがって行うことも可能です。モデルの反復ごとに、損失関数が計算されます。hyperoptは、さまざまな最適化アルゴリズムを用いて探索空間をナビゲートし、損失関数が返す値を最小化するために利用可能な最適なパラメータ設定の組合せを見つけます。
# MAGIC 
# MAGIC To make use of hyperopt, lets define our search space and re-write our model training and evaluation logic to provide a single function call which will return a loss function measure:
# MAGIC 
# MAGIC hyperoptを利用するために、探索空間を定義し、モデルの学習と評価のロジックを書き換えて、損失関数の測定を返す単一の関数呼び出しを提供することにします。

# COMMAND ----------

from hyperopt import hp, fmin, tpe, rand, SparkTrials, STATUS_OK, STATUS_FAIL, space_eval

# define search space
search_space = hp.choice('model_type',[
                  {'type':'Pareto/NBD', 'l2':hp.uniform('pareto_nbd_l2', 0.0, 1.0)},
                  {'type':'BG/NBD'    , 'l2':hp.uniform('bg_nbd_l2', 0.0, 1.0)}  
                  ]
                )

# define function for model evaluation
def evaluate_model(params):
  
  # accesss replicated input_pd dataframe
  data = inputs.value
  
  # retrieve incoming parameters
  model_type = params['type']
  l2_reg = params['l2']
  
  # instantiate and configure the model
  if model_type == 'BG/NBD':
    model = BetaGeoFitter(penalizer_coef=l2_reg)
  elif model_type == 'Pareto/NBD':
    model = ParetoNBDFitter(penalizer_coef=l2_reg)
  else:
    return {'loss': None, 'status': STATUS_FAIL}
  
  # fit the model
  model.fit(data['frequency_cal'], data['recency_cal'], data['T_cal'])
  
  # evaluate the model
  frequency_holdout_actual = data['frequency_holdout']
  frequency_holdout_predicted = model.predict(data['duration_holdout'], data['frequency_cal'], data['recency_cal'], data['T_cal'])
  mse = score_model(frequency_holdout_actual, frequency_holdout_predicted, 'mse')
  
  # return score and status
  return {'loss': mse, 'status': STATUS_OK}

# COMMAND ----------

# MAGIC %md Notice that the evaluate_model function retrieves its data from a variable named inputs.  Inputs is defined in the next cell as a [broadcast variable](https://spark.apache.org/docs/latest/rdd-programming-guide.html#broadcast-variables) containing the inputs_pd DataFrame used earlier. As a broadcast variable, a complete stand-alone copy of the dataset used by the model is replicated to each worker in the Spark cluster.  This limits the amount of data that must be sent from the cluster driver to the workers with each hyperopt iteration.  For more information on this and other hyperopt best practices, please refer to [this document](https://docs.databricks.com/applications/machine-learning/automl/hyperopt/hyperopt-best-practices.html).
# MAGIC 
# MAGIC evaluate_model 関数は inputs という名前の変数からデータを取得していることに注意してください。inputsは次のセルで、先ほど使用したinputs_pd DataFrameを含むブロードキャスト変数として定義されています。ブロードキャスト変数として、モデルで使用されるデータセットの完全なスタンドアロンコピーがSparkクラスタの各ワーカーにレプリケートされます。これにより、hyperoptのイテレーションごとにクラスタードライバーからワーカーに送らなければならないデータ量を制限することができます。これと他の hyperopt のベストプラクティスの詳細については、このドキュメントを参照してください。
# MAGIC 
# MAGIC With everything in place, let's perform our hyperparameter tuning over 100 iterations in order to identify the best model type and L2 settings for our dataset:
# MAGIC 
# MAGIC 準備が整ったところで、100回の繰り返しでハイパーパラメータのチューニングを行い、データセットに最適なモデルの種類とL2設定を特定しましょう。

# COMMAND ----------

import mlflow

# replicate input_pd dataframe to workers in Spark cluster
inputs = sc.broadcast(input_pd)

# configure hyperopt settings to distribute to all executors on workers
spark_trials = SparkTrials(parallelism=2)

# select optimization algorithm
algo = tpe.suggest

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

# MAGIC %md When used with the Databricks ML runtime, the individual runs that make up the search space evaluation are tracked in a built-in repository called mlflow. For more information on how to review the models generated by hyperopt using the Databricks mlflow interface, please check out [this document](https://docs.databricks.com/applications/machine-learning/automl/hyperopt/hyperopt-spark-mlflow-integration.html).
# MAGIC 
# MAGIC Databricks MLランタイムと共に使用する場合、探索空間評価を構成する個々の実行はmlflowというビルトインリポジトリで追跡されます。Databricksのmlflowインターフェイスを使用してhyperoptが生成したモデルをレビューする方法の詳細については、このドキュメントを参照してください。
# MAGIC 
# MAGIC The optimal hyperparameter settings observed during the hyperopt iterations are captured in the argmin variable.  Using the space_eval function, we can obtain a friendly representation of which settings performed best:
# MAGIC 
# MAGIC hyperoptの反復処理中に観測された最適なハイパーパラメータ設定は、argmin変数に格納されます。space_eval関数を使用すると、どの設定が最適であったかを分かりやすく表現することができます。

# COMMAND ----------

# print optimum hyperparameter settings
print(space_eval(search_space, argmin))

# COMMAND ----------

# MAGIC %md Now that we know our *best* parameter settings, let's train the model with these to enable us to perform some more in-depth model evaluation:
# MAGIC 
# MAGIC 最適なパラメータ設定がわかったので、このパラメータでモデルを学習させ、より詳細なモデル評価を行えるようにしましょう。
# MAGIC 
# MAGIC NOTE Because of how search spaces are *searched*, different hyperopt runs may yield slightly different results. 
# MAGIC 
# MAGIC 注：探索空間がどのように探索されるかによって、Hyperoptの実行結果は若干異なる場合があります。

# COMMAND ----------

# get hyperparameter settings
params = space_eval(search_space, argmin)
model_type = params['type']
l2_reg = params['l2']

# instantiate and configure model
if model_type == 'BG/NBD':
  model = BetaGeoFitter(penalizer_coef=l2_reg)
elif model_type == 'Pareto/NBD':
  model = ParetoNBDFitter(penalizer_coef=l2_reg)
else:
  raise 'Unrecognized model type'
  
# train the model
model.fit(input_pd['frequency_cal'], input_pd['recency_cal'], input_pd['T_cal'])

# COMMAND ----------

# MAGIC %md ###Step 5: Evaluate the Model
# MAGIC 
# MAGIC ###ステップ5：モデルの評価
# MAGIC 
# MAGIC Using a method defined in the last section of this notebook, we can calculate the MSE for our newly trained model:
# MAGIC 
# MAGIC このノートブックの最後のセクションで定義された方法を使用して、新しく学習したモデルのMSEを計算することができます。

# COMMAND ----------

# score the model
frequency_holdout_actual = input_pd['frequency_holdout']
frequency_holdout_predicted = model.predict(input_pd['duration_holdout'], input_pd['frequency_cal'], input_pd['recency_cal'], input_pd['T_cal'])
mse = score_model(frequency_holdout_actual, frequency_holdout_predicted, 'mse')

print('MSE: {0}'.format(mse))

# COMMAND ----------

# MAGIC %md While important for comparing models, the MSE metric is a bit more challenging to interpret in terms of the overall goodness of fit of any individual model.  To provide more insight into how well our model fits our data, let's visualize the relationships between some actual and predicted values.
# MAGIC 
# MAGIC MSEはモデルの比較には重要ですが、個々のモデルの全体的な適合度という観点からは、もう少し解釈が難しい指標です。モデルがデータにどの程度適合しているかをより深く理解するために、いくつかの実際値と予測値の関係を可視化してみましょう。
# MAGIC 
# MAGIC To get started, we can examine how purchase frequencies in the calibration period relates to actual (frequency_holdout) and predicted (model_predictions) frequencies in the holdout period:
# MAGIC 
# MAGIC まず、キャリブレーション期間中の購入頻度と、ホールドアウト期間中の実績（frequency_holdout）および予測（model_predictions）頻度がどのように関連しているのかを調べます。

# COMMAND ----------

from lifetimes.plotting import plot_calibration_purchases_vs_holdout_purchases

plot_calibration_purchases_vs_holdout_purchases(
  model, 
  input_pd, 
  n=90, 
  **{'figsize':(8,8)}
  )

display()

# COMMAND ----------

# MAGIC %md What we see here is that a higher number of purchases in the calibration period predicts a higher average number of purchases in the holdout period but the actual values diverge sharply from model predictions when we consider customers with a large number of purchases (>60) in the calibration period.  Thinking back to the charts in the data exploration section of this notebook, you might recall that there are very few customers with such a large number of purchases so that this divergence may be a result of a very limited number of instances at the higher end of the frequency range. More data may bring the predicted and actuals back together at this higher end of the curve.  If this divergence persists, it may indicate a range of customer engagement frequency above which we cannot make reliable predictions.
# MAGIC 
# MAGIC ここでわかるのは、キャリブレーション期間の購入回数が多いほど、ホールドアウト期間の平均購入回数が多くなることが予測されるが、キャリブレーション期間の購入回数が多い（60回以上）顧客を考慮すると、実際の値はモデルの予測値から大きく乖離するということである。このノートのデータ探索セクションのチャートを思い起こすと、このような購入回数の多い顧客は非常に少ないので、この乖離は、頻度範囲の高い方のインスタンスの数が非常に少ない結果である可能性があることを思い出すかもしれません。より多くのデータによって、この高い方のカーブで予測値と実績値が一致する可能性があります。この乖離が続く場合、信頼できる予測ができない顧客エンゲージメント頻度の範囲を示している可能性があります。
# MAGIC 
# MAGIC Using the same method call, we can visualize time since last purchase relative to the average number of purchases in the holdout period. This visualization illustrates that as time since the last purchase increases, the number of purchases in the holdout period decreases.  In otherwords, those customers we haven't seen in a while aren't likely coming back anytime soon:
# MAGIC 
# MAGIC 同じメソッドを使って、最後の購入からの時間と、保留期間中の平均購入回数の関係を可視化することができます。この可視化では、最終購入からの時間が長くなるにつれて、保留期間中の購入回数が減少することが示されている。つまり、しばらく会っていない顧客は、すぐには戻ってこない可能性が高いのです。
# MAGIC 
# MAGIC NOTE As before, we will hide the code in the following cells to focus on the visualizations.  Use **Show code** to see the associated Python logic.
# MAGIC 
# MAGIC 注 前回と同様、ビジュアライゼーションに集中するために、以下のセル内のコードを非表示にします。関連するPythonのロジックを見るには、Show codeを使用してください。

# COMMAND ----------

plot_calibration_purchases_vs_holdout_purchases(
  model, 
  input_pd, 
  kind='time_since_last_purchase', 
  n=90, 
  **{'figsize':(8,8)}
  )

display()

# COMMAND ----------

# MAGIC %md Plugging the age of the customer at the time of the last purchase into the chart shows that the timing of the last purchase in a customer's lifecycle doesn't seem to have a strong influence on the number of purchases in the holdout period until a customer becomes quite old.  This would indicate that the customers that stick around a long while are likely to be more frequently engaged:
# MAGIC 
# MAGIC 最後に購入した時の年齢をグラフに突っ込んでみると、顧客のライフサイクルにおける最後の購入のタイミングは、かなり高齢になるまで、保有期間の購入回数に強い影響を与えないようであることがわかる。このことは、長く付き合っている顧客ほど、エンゲージメントの頻度が高い可能性があることを示しているのだろう。

# COMMAND ----------

plot_calibration_purchases_vs_holdout_purchases(
  model, 
  input_pd, 
  kind='recency_cal', 
  n=300,
  **{'figsize':(8,8)}
  )

display()

# COMMAND ----------

# MAGIC %md From a quick visual inspection, it's fair to say our model isn't perfect but there are some useful patterns that it captures. Using these patterns, we might calculate the probability a customer remains engaged:
# MAGIC 
# MAGIC ざっと見たところ、このモデルは完璧ではありませんが、いくつかの有用なパターンを捉えていると言えるでしょう。これらのパターンを使って、顧客がエンゲージメントを維持する確率を計算することができます。

# COMMAND ----------

# add a field with the probability a customer is currently "alive"
filtered_pd['prob_alive']=model.conditional_probability_alive(
    filtered_pd['frequency'], 
    filtered_pd['recency'], 
    filtered_pd['T']
    )

filtered_pd.head(10)

# COMMAND ----------

# MAGIC %md The prediction of the customer's probability of being alive could be very interesting for the application of the model to our marketing processes.  But before exploring model deployment, let's take a look at how this probability changes as customers re-engage by looking at the history of a single customer with modest activity in the dataset, CustomerID 12383:
# MAGIC 
# MAGIC 顧客の生存確率の予測は、モデルを我々のマーケティングプロセスに応用する上で非常に興味深いものになる可能性があります。しかし、モデルの展開を探る前に、データセットの中で控えめな活動をする一人の顧客、CustomerID 12383の履歴を見ることで、顧客が再係合するとこの確率がどのように変化するかを見てみよう。

# COMMAND ----------

from lifetimes.plotting import plot_history_alive
import matplotlib.pyplot as plt

# clear past visualization instructions
plt.clf()
plt.subplots(figsize=(12, 8))

# customer of interest
CustomerID = '12383'

# grab customer's metrics and transaction history
cmetrics_pd = input_pd[input_pd['CustomerID']==CustomerID]
trans_history = orders_pd.loc[orders_pd['CustomerID'] == CustomerID]

# calculate age at end of dataset
days_since_birth = 400

# plot history of being "alive"
plot_history_alive(
  model, 
  days_since_birth, 
  trans_history, 
  'InvoiceDate'
  )

display()

# COMMAND ----------

# MAGIC %md From this chart, we can see this customer made his or her first purchase in January 2011 followed by a repeat purchase later that month.  There was about a 1-month lull in activity during which the probability of the customer being alive declined slightly but with purchases in March, April and June of that year, the customer sent repeated signals that he or she was engaged. Since that last June purchase, the customer hasn't been seen in our transaction history, and our belief that the customer remains engaged has been dropping though as a moderate pace given the signals previously sent.
# MAGIC 
# MAGIC このグラフから、この顧客は2011年1月に初めて購入し、同月末にリピート購入したことがわかる。その後、1ヶ月ほど停滞し、生存確率は若干低下しましたが、3月、4月、6月に購入され、生存していることを示すシグナルを何度も発信されています。この6月の購入以降、そのお客様は取引履歴に現れておらず、以前のシグナルを考えると、お客様が生存している確率は緩やかなペースではありますが、下がり続けています。
# MAGIC 
# MAGIC How does the model arrive at these probabilities? The exact math is tricky but by plotting the probability of being alive as a heatmap relative to frequency and recency, we can understand the probabilities assigned to the intersections of these two values:
# MAGIC 
# MAGIC この確率は、どのようにして算出されるのでしょうか？正確な計算は難しいが、生存している確率を頻度と再来店の相対的なヒートマップとしてプロットすることで、この2つの値の交点に割り当てられた確率を理解することができる。

# COMMAND ----------

from lifetimes.plotting import plot_probability_alive_matrix

# set figure size
plt.subplots(figsize=(12, 8))

plot_probability_alive_matrix(model)

display()

# COMMAND ----------

# MAGIC %md In addition to predicting the probability a customer is still alive, we can calculate the number of purchases expected from a customer over a given future time interval, such as over the next 30-days:
# MAGIC 
# MAGIC 顧客が生存している確率を予測するだけでなく、今後30日間といった将来の時間間隔において、顧客から予想される購入数を計算することができる。

# COMMAND ----------

from lifetimes.plotting import plot_frequency_recency_matrix

# set figure size
plt.subplots(figsize=(12, 8))

plot_frequency_recency_matrix(model, T=30)

display()

# COMMAND ----------

# MAGIC %md As before, we can calculate this probability for each customer based on their current metrics:
# MAGIC 
# MAGIC 先ほどと同様に、この確率は各顧客の現在の指標に基づいて計算することができます。

# COMMAND ----------

filtered_pd['purchases_next30days']=(
  model.conditional_expected_number_of_purchases_up_to_time(
    30, 
    filtered_pd['frequency'], 
    filtered_pd['recency'], 
    filtered_pd['T']
    )
  )

filtered_pd.head(10)

# COMMAND ----------

# MAGIC %md ###Step 6: Deploy the Model for Predictions
# MAGIC 
# MAGIC ###ステップ6：予測用モデルの展開
# MAGIC 
# MAGIC There are numerous ways we might make use of the trained BTYD model.  We may wish to understand the probability a customer is still engaged.  We may also wish to predict the number of purchases expected from the customer over some number of days.  All we need to make these predictions is our trained model and values of frequency, recency and age (T) for the customer as demonstrated here:
# MAGIC 
# MAGIC 学習させたBTYDモデルの利用方法は数多くあります。顧客がまだ関与している確率を理解したい場合があります。また、ある日数の間に顧客から予想される購入数を予測したいかもしれません。これらの予測に必要なのは、学習したモデルと、ここで示したような顧客の頻度、再来店、年齢（T）の値だけです。

# COMMAND ----------

frequency = 6
recency = 255
T = 300
t = 30

print('Probability of Alive: {0}'.format( model.conditional_probability_alive(frequency, recency, T) ))
print('Expected Purchases in next {0} days: {1}'.format(t, model.conditional_expected_number_of_purchases_up_to_time(t, frequency, recency, T) ))

# COMMAND ----------

# MAGIC %md The challenge now is to package our model into something we could re-use for this purpose. Earlier, we used mlflow in combination with hyperopt to capture model runs during the hyperparameter tuning exercise. As a platform, mlflow is designed to solve a wide range of challenges that come with model development and deployment, including the deployment of models as functions and microservice applications. 
# MAGIC 
# MAGIC 今の課題は、この目的のために再利用できるようなモデルにパッケージすることです。先に、ハイパーパラメータのチューニングを行う際に、mlflowとhyperoptを組み合わせて、モデルの実行をキャプチャしました。プラットフォームとしてのmlflowは、関数やマイクロサービス・アプリケーションとしてのモデルのデプロイメントなど、モデルの開発とデプロイメントに伴う幅広い課題を解決するために設計されています。
# MAGIC 
# MAGIC MLFlow tackles deployment challenges out of the box for a number of [popular model types](https://www.mlflow.org/docs/latest/models.html#built-in-model-flavors). However, lifetimes models are not one of these. To use mlflow as our deployment vehicle, we need to write a custom wrapper class which translates the standard mlflow API calls into logic which can be applied against our model.
# MAGIC 
# MAGIC MLFlowは、多くの一般的なモデルタイプについて、デプロイの課題にすぐに取り組むことができます。しかし、ライフタイムモデルは、これらのうちの1つではありません。デプロイメント手段としてmlflowを使用するには、標準的なmlflow APIコールをモデルに対して適用できるロジックに変換するカスタムラッパークラスを書く必要があります。
# MAGIC 
# MAGIC To illustrate this, we've implemented a wrapper class for our lifetimes model which maps the mlflow *predict()* method to multiple prediction calls against our model.  Typically, we'd map *predict()* to a single prediction but we've bumped up the complexity of the returned result to show one of many ways the wrapper may be employed to implement custom logic:
# MAGIC 
# MAGIC これを説明するために、mlflowのpredict()メソッドをモデルに対する複数の予測呼び出しにマッピングするlifetimesモデルのラッパークラスを実装してみました。通常、predict()は単一の予測にマップされますが、カスタムロジックを実装するためにラッパーを使用する多くの方法の一つを示すために、返される結果の複雑さを増加させました。

# COMMAND ----------

import mlflow
import mlflow.pyfunc

# create wrapper for lifetimes model
class _lifetimesModelWrapper(mlflow.pyfunc.PythonModel):
  
    def __init__(self, lifetimes_model):
        self.lifetimes_model = lifetimes_model

    def predict(self, context, dataframe):
      
      # access input series
      frequency = dataframe.iloc[:,0]
      recency = dataframe.iloc[:,1]
      T = dataframe.iloc[:,2]
      
      # calculate probability currently alive
      results = pd.DataFrame( 
                  self.lifetimes_model.conditional_probability_alive(frequency, recency, T),
                  columns=['alive']
                  )
      # calculate expected purchases for provided time period
      results['purch_15day'] = self.lifetimes_model.conditional_expected_number_of_purchases_up_to_time(15, frequency, recency, T)
      results['purch_30day'] = self.lifetimes_model.conditional_expected_number_of_purchases_up_to_time(30, frequency, recency, T)
      results['purch_45day'] = self.lifetimes_model.conditional_expected_number_of_purchases_up_to_time(45, frequency, recency, T)
      
      return results[['alive', 'purch_15day', 'purch_30day', 'purch_45day']]

# COMMAND ----------

# MAGIC %md We now need to register our model with mlflow. As we do this, we inform it of the wrapper that maps its expected API to the model's functionality.  We also provide environment information to instruct it as to which libraries it needs to install and load for our model to work:
# MAGIC 
# MAGIC ここで、モデルをmlflowに登録する必要があります。これを行う際に、期待されるAPIをモデルの機能にマッピングするラッパーを通知します。また、モデルが動作するためにどのライブラリをインストールし、ロードする必要があるかを指示するために、環境情報を提供します。
# MAGIC 
# MAGIC NOTE We would typically train and log our model as a single step but in this notebook we've separated the two actions in order to focus here on custom model deployment.  For examine more common patterns of mlflow implementation, please refer to [this](https://docs.databricks.com/applications/mlflow/model-example.html) and other examples available online. 
# MAGIC 
# MAGIC 注：通常、モデルの学習とログ取得は1つのステップとして行われますが、このノートブックでは、カスタムモデルのデプロイメントに焦点を当てるために、2つのアクションを分離しました。より一般的なmlflowの実装パターンについては、こちらやオンラインで利用可能な他の例を参照してください。

# COMMAND ----------

# add lifetimes to conda environment info
conda_env = mlflow.pyfunc.get_default_conda_env()
# conda_env['dependencies'][1]['pip'] += ['lifetimes==0.10.1'] # version should match version installed at top of this notebook

# save model run to mlflow
with mlflow.start_run(run_name='deployment run') as run:
  mlflow.pyfunc.log_model(
    'model', 
    python_model=_lifetimesModelWrapper(model), 
    conda_env=conda_env
    )

# COMMAND ----------

# MAGIC %md Now that our model along with its dependency information and class wrapper have been recorded, let's use mlflow to convert the model into a function we can employ against a Spark DataFrame:
# MAGIC 
# MAGIC モデル、依存情報、クラスラッパーが記録されたので、mlflowを使ってモデルをSpark DataFrameに対して使える関数に変換しましょう。

# COMMAND ----------

from pyspark.sql.types import ArrayType, FloatType

# define the schema of the values returned by the function
result_schema = ArrayType(FloatType())

# define function based on mlflow recorded model
probability_alive_udf = mlflow.pyfunc.spark_udf(
  spark, 
  'runs:/{0}/model'.format(run.info.run_id), 
  result_type=result_schema
  )

# register the function for use in SQL
_ = spark.udf.register('probability_alive', probability_alive_udf)

# COMMAND ----------

# MAGIC %md Assuming we had access to customer metrics for frequency, recency and age, we can now use our function to generate some predictions:
# MAGIC 
# MAGIC 頻度、再帰性、年齢などの顧客指標にアクセスできたと仮定して、この関数を使っていくつかの予測を生成することができます。

# COMMAND ----------

# create a temp view for SQL demonstration (next cell)
filtered.createOrReplaceTempView('customer_metrics')

# demonstrate function call on Spark DataFrame
display(
  filtered
    .withColumn(
      'predictions', 
      probability_alive_udf(filtered.frequency, filtered.recency, filtered.T)
      )
    .selectExpr(
      'customerid', 
      'predictions[0] as prob_alive', 
      'predictions[1] as purch_15day', 
      'predictions[2] as purch_30day', 
      'predictions[3] as purch_45day'
      )
  )

# COMMAND ----------

# MAGIC %sql -- predict probabiliies customer is alive and will return in 15, 30 & 45 days
# MAGIC 
# MAGIC SELECT
# MAGIC   x.CustomerID,
# MAGIC   x.prediction[0] as prob_alive,
# MAGIC   x.prediction[1] as purch_15day,
# MAGIC   x.prediction[2] as purch_30day,
# MAGIC   x.prediction[3] as purch_45day
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC     CustomerID,
# MAGIC     probability_alive(frequency, recency, T) as prediction
# MAGIC   FROM customer_metrics
# MAGIC   ) x;

# COMMAND ----------

# MAGIC %md With our lifetimes model now registered as a function, we can incorporate customer lifetime probabilities into our ETL batch, streaming and interactive query workloads.  Levering additional model deployment capabilities of mlflow, our model may also be deployed as a standalone webservice leveraging [AzureML](https://docs.azuredatabricks.net/_static/notebooks/mlflow/mlflow-quick-start-deployment-azure.html) and [AWS Sagemaker](https://docs.databricks.com/_static/notebooks/mlflow/mlflow-quick-start-deployment-aws.html). 
# MAGIC 
# MAGIC ライフタイムモデルが関数として登録されたことで、顧客のライフタイム確率をETLのバッチ処理、ストリーミング処理、インタラクティブなクエリー処理に取り入れることができるようになりました。また、mlflowのモデル展開機能を活用し、AzureMLやAWS Sagemakerを利用したスタンドアローンWebサービスとして展開することも可能です。
