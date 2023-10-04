# Databricks notebook source
import os

from pyspark.sql import DataFrame
from pyspark.sql.types import StructType, StructField, StringType, FloatType, TimestampType
from pyspark.sql import functions as F
from pyspark.sql.window import Window

import pandas as pd

import seaborn

csv_url = "https://zenodo.org/record/4624805/files/NP.csv"
df = spark.createDataFrame(
    pd.read_csv(csv_url),
    schema=StructType([
        StructField("Date String", StringType(), True),
        StructField("Price", FloatType(), True),
        StructField("Grid load forecast", FloatType(), True),
        StructField("Wind power forecast", FloatType(), True),
    ])
)
df = df.withColumn("Timestamp", F.col("Date String").cast(TimestampType()))

display(df)

# COMMAND ----------

df = df.withColumn(
    "Rolling Weekly Price", 
    F.avg("Price").over(
        Window
        .orderBy(F.col("Timestamp").cast('long'))
        .rangeBetween(-7*24*60*60, 0)
    ),
)
df = df.withColumn(
    "Rolling Daily Price", 
    F.avg("Price").over(
        Window
        .orderBy(F.col("Timestamp").cast('long'))
        .rangeBetween(-24*60*60, 0)
    ),
)
df = df.withColumn(
    "Rolling 4h Price", 
    F.avg("Price").over(
        Window
        .orderBy(F.col("Timestamp").cast('long'))
        .rangeBetween(-4*60*60, 0)
    ),
)
df = df.withColumn(
    "Next Price", 
    F.avg("Price").over(
        Window
        .orderBy(F.col("Timestamp").cast('long'))
        .rangeBetween(1,60*60)
    ),
)
df = df.withColumn(
    "Prev Price", 
    F.avg("Price").over(
        Window
        .orderBy(F.col("Timestamp").cast('long'))
        .rangeBetween(-60*60,-1)
    ),
)
display(df.tail(1000))

# COMMAND ----------

tmp_df = df[
    (df["Timestamp"] >= F.lit("2016-01-01").cast(TimestampType())) &
    (df["Timestamp"] < F.lit("2016-02-01").cast(TimestampType()))
]
tmp_df = tmp_df.toPandas()

import matplotlib.pyplot as plt

plt.figure()
seaborn.scatterplot(x=tmp_df["Next Price"], y=tmp_df["Rolling Weekly Price"])
plt.figure()
seaborn.scatterplot(x=tmp_df["Next Price"], y=tmp_df["Rolling Daily Price"])
plt.figure()
seaborn.scatterplot(x=tmp_df["Next Price"], y=tmp_df["Rolling 4h Price"])
plt.figure()
seaborn.scatterplot(x=tmp_df["Next Price"], y=tmp_df["Price"])

# COMMAND ----------

from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler

lr_df = df.na.drop()

vectorAssembler = VectorAssembler(outputCol="features")
vectorAssembler.setInputCols(["Price", "Prev Price"]) #"Rolling 4h Price", "Rolling Daily Price", "Rolling Weekly Price"])
lr_df = vectorAssembler.transform(lr_df)

lr = LinearRegression(labelCol="Next Price", featuresCol="features")
model = lr.fit(lr_df)
lr_df_sq = model.transform(lr_df)

display(lr_df_sq)

lr_huber = LinearRegression(labelCol="Next Price", featuresCol="features", loss="huber")
model = lr_huber.fit(lr_df)
lr_df_huber = model.transform(lr_df)

display(lr_df_huber)

# COMMAND ----------

import matplotlib.pyplot as plt

tmp_df = lr_df_sq[
    (lr_df["Timestamp"] >= F.lit("2016-01-01").cast(TimestampType())) &
    (lr_df["Timestamp"] < F.lit("2016-02-01").cast(TimestampType()))
]
tmp_df_sq = tmp_df.toPandas()

tmp_df = lr_df_huber[
    (lr_df["Timestamp"] >= F.lit("2016-01-01").cast(TimestampType())) &
    (lr_df["Timestamp"] < F.lit("2016-02-01").cast(TimestampType()))
]
tmp_df_huber = tmp_df.toPandas()

fig, ax = plt.subplots()
seaborn.scatterplot(x=tmp_df_sq["Next Price"], y=tmp_df_sq["prediction"], ax=ax)
seaborn.scatterplot(x=tmp_df_huber["Next Price"], y=tmp_df_huber["prediction"], ax=ax)

fig, ax = plt.subplots()
seaborn.scatterplot(x=tmp_df_sq["Next Price"], y=tmp_df_sq["prediction"], ax=ax)
seaborn.scatterplot(x=tmp_df_huber["Next Price"], y=tmp_df_huber["prediction"], ax=ax)
ax.set_xlim(0,50)
ax.set_ylim(0,50)


# COMMAND ----------


