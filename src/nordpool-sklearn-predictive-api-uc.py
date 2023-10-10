# Databricks notebook source
# MAGIC %md
# MAGIC # Nordpool sklearn predictive API - Unity catalog
# MAGIC
# MAGIC See [README.md](../README.md) for description.
# MAGIC

# COMMAND ----------

# MAGIC %pip install "mlflow-skinny[databricks]>=2.4.1"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Read and plot data
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

# read
# original source csv_url = "https://zenodo.org/record/4624805/files/NP.csv"
csv_url = "file:../data/NP.csv"
df = pd.read_csv(csv_url) # read csv
df = df.rename(columns=(lambda col_name: col_name.strip())) # clean up column names
df.display()

# plot full data
fig = plt.figure()
sns.lineplot(x=pd.to_datetime(df["Date"]), y=df["Price"], linewidth=.5)

# plot the last month
N_plot = 30*24
fig = plt.figure()
sns.lineplot(x=pd.to_datetime(df["Date"])[-N_plot:], y=df["Price"][-N_plot:], linewidth=1)
plt.gca().xaxis.set_major_locator(matplotlib.dates.DayLocator(interval=7)) # every 7th date


# COMMAND ----------

# Standard imports

# python imports
from dataclasses import dataclass
from typing import List, Tuple, Dict

# sklearn pipeline
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import TheilSenRegressor

# mlflow
import mlflow
from mlflow.models import infer_signature
from mlflow.models import ModelSignature, infer_signature
from mlflow.types.schema import Schema, ColSpec, TensorSpec


# COMMAND ----------

# Example of an API call from a tutorial
"""
  {
    "dataframe_split": [{
      "index": [0, 1],
      "columns": ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"],
      "data": [[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2]]
    }]
  }
"""

# COMMAND ----------

test_df = pd.DataFrame({
    "Date": [
        [
            "2013-01-01 00:00:00",
            "2013-01-01 01:00:00",
            "2013-01-01 02:00:00",
            "2013-01-01 03:00:00",
            "2013-01-01 04:00:00",
        ],
        [
            "2013-01-01 00:00:00",
            "2013-01-02 01:00:00",
            "2013-01-03 02:00:00",
            "2013-01-03 03:00:00",
            "2013-01-03 04:00:00",
        ],
    ],
    "Price": [
        [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
        ],
        [
            10.0,
            20.0,
            30.0,
            40.0,
            50.0,
        ],
    ]
})

# COMMAND ----------

# TimedeltaInterval describe time interval relative to some reference time, 
# typically the latest time in time-series
@dataclass
class TimedeltaInterval:
    beg: pd.Timedelta
    end: pd.Timedelta

# COMMAND ----------

# The time-intervals we are interested in
timedelta_intervals = {
    "Current time": TimedeltaInterval(pd.Timedelta("-1s"), pd.Timedelta("1s")),
    "Last hour": TimedeltaInterval(pd.Timedelta("-1h"), pd.Timedelta("0s")),
    "Last 4 hours": TimedeltaInterval(pd.Timedelta("-4h"), pd.Timedelta("0s")),
    "Last day": TimedeltaInterval(pd.Timedelta("-1d"), pd.Timedelta("0s")),
    "Last week": TimedeltaInterval(pd.Timedelta("-7d"), pd.Timedelta("0s"))
}


# COMMAND ----------

@dataclass
class LatestTimeIntervalAverageTransformer(BaseEstimator, TransformerMixin):
    """Transforms a time-series to sets of time-intervals relative to the latest time in the input time-series
    
    For example, takes two weeks of Time-Price pairs and converts to Last week, Last day, Last 4 hour average prices, and Last and Current hour prices."""
    timedelta_intervals: Dict[str,TimedeltaInterval]

    def _interval_mean_price(
        self,
        timestamps: List[pd.Timestamp], 
        prices: List[float], 
        ref_time: pd.Timestamp, 
        timedelta_interval: TimedeltaInterval,
    ) -> float:
        """Calculates mean price for an time interval relative to a reference time"""
        df = pd.DataFrame({ "Timestamp": timestamps, "Price": prices })
        df = df[(
            (df["Timestamp"] >= ref_time + timedelta_interval.beg) &
            (df["Timestamp"] < ref_time + timedelta_interval.end)
        )]
        return df["Price"].mean()

    def transform(self, X: pd.DataFrame, y = None):
        """Transforms the list of input time-series to a list of time-intervals."""
        # X = N x (datetime64, float)
        # find the latest timestamp
        Xt = X.copy(deep=True)
        Xt["Timestamp"] = X.apply(
            lambda row : pd.to_datetime(row["Date"]).max(), 
            axis = "columns"
        )

        # find rows that are between timedeltas with respect to the latest timestamp
        for col_name, timedelta_interval in self.timedelta_intervals.items():
            Xt[col_name] = Xt.apply(
                lambda row : self._interval_mean_price(
                    pd.to_datetime(row["Date"]),
                    row["Price"],
                    row["Timestamp"],
                    timedelta_interval,
                ),
                axis = "columns",
            )
        
        return Xt.drop(columns=["Timestamp", "Date", "Price"])
    
    def fit(self, X, y=None):
        """Fit is pass-through for transformers"""
        return self



# COMMAND ----------

# pipeline with preprocessor and Theil Sen regressor (similar to linear regression)
pipeline = Pipeline(steps=[
    ("latest_time_interval_avgs", LatestTimeIntervalAverageTransformer(timedelta_intervals)),
    ("theil_sen_regressor", TheilSenRegressor())
])

# COMMAND ----------

# training params
N_window = 8*24  # 8 days of training data with 1h frequency, only the last week is used
N_samples = 1000 # 1000 random samples instead of the whole dataset

# construct training data
train_df = df.copy(deep=True)
windows = train_df.rolling(N_window)

display(windows)

# take only windows that have full length
windows = [window for window in windows if len(window) >= N_window]

# take N_samples samples from input and target data
# X = list of input time-series, N_samples x N_window x (Date, Price)
# y = lists of targets, N_samples x Next Price
Xy = pd.DataFrame({
    "Date": [window["Date"].values[:-1] for window in windows],
    "Price": [window["Price"].values[:-1] for window in windows],
    "Next Price": [window["Price"].values[-1] for window in windows]
}).sample(N_samples)

display(Xy.shape)

X = Xy[["Date","Price"]]
y = Xy["Next Price"]


# COMMAND ----------

# enable MLflow autologging for model metrics
mlflow.sklearn.autolog()

# start MLflow run
with mlflow.start_run() as run:
    model = pipeline.fit(X,y)
    model_info = mlflow.sklearn.log_model(
        sk_model=model, 
        artifact_path="model", # ToDo: use unity catalog?
        input_example=X.head(2),
    )
# model_from_registry
sklearn_pyfunc = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)

results = pd.DataFrame({
    "Next Price": y,
    "Saved Model": model.predict(X),
    "Loaded Model": sklearn_pyfunc.predict(X),
})
display(results)

# COMMAND ----------

pred_windows = windows[-N_plot:]
df_pred = pd.DataFrame({
    "Date": [window["Date"].values[:-1] for window in pred_windows],
    "Price": [window["Price"].values[:-1] for window in pred_windows],
})
df_pred["Pred Price"] = model.predict(df_pred)
df_pred["Next Price"] = [window["Price"].values[-1] for window in pred_windows]
df_pred["Last Date"] = [window["Date"].values[-1] for window in pred_windows]

df_pred.display()

fig = plt.figure()
sns.lineplot(x=pd.to_datetime(df_pred["Last Date"])[-N_plot:], y=df_pred["Next Price"][-N_plot:], linewidth=1)
sns.lineplot(x=pd.to_datetime(df_pred["Last Date"])[-N_plot:], y=df_pred["Pred Price"][-N_plot:], linewidth=1)
plt.xlabel("Date")
plt.ylabel("Price vs Fitted")
plt.gca().xaxis.set_major_locator(matplotlib.dates.DayLocator(interval=7)) # every 7th date


# COMMAND ----------

# NOTE: Skip this if using Unity Catalog and run the next cell
#
# Go to 'Experiments' tab and select this experiment 'nordpool-sklearn-predictive-api'
# Select this run (the latest?) and click the model and examine it
# Press 'Register model', select Workspace Model Registry and 'nordpool' model
# Go to 'Serving' tab and select 'nordpool' endpoint
# Click 'Edit configuration' and update the version 


# COMMAND ----------

# Add model to Unity Catalog
import mlflow
catalog = "ml_example"
schema = "ml_models"
model_name = "nordpool"
mlflow.set_registry_uri("databricks-uc")
mlflow.register_model(
    model_uri=f"runs:/{model_info.run_id}/model",
    name=f"{catalog}.{schema}.{model_name}"
)
