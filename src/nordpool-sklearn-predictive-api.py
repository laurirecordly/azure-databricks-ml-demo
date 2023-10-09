# Databricks notebook source
# Kuvaaja
# Unity Catalog
# Siivous
# Nimi
# Autolog
# pip install git


# COMMAND ----------

# MAGIC %pip install mlflow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd

#csv_url = "https://zenodo.org/record/4624805/files/NP.csv"
csv_url = "file:../data/NP.csv"
df = pd.read_csv(csv_url)
df = df.rename(columns=(lambda col_name: col_name.strip()))
display(df)

# COMMAND ----------

# Hide imports in own cell
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from sklearn.linear_model import TheilSenRegressor

import mlflow
from mlflow.models import infer_signature
from mlflow.models import ModelSignature, infer_signature
from mlflow.types.schema import Schema, ColSpec, TensorSpec

"""
  {
    "dataframe_split": [{
      "index": [0, 1],
      "columns": ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"],
      "data": [[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2]]
    }]
  }
"""

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



@dataclass
class TimedeltaInterval:
    beg: pd.Timedelta
    end: pd.Timedelta

@dataclass
class LatestTimeIntervalAverageTransformer(BaseEstimator, TransformerMixin):
    timedelta_intervals: Dict[str,TimedeltaInterval]
                         
    def fit(self, X, y=None):
        return self

    def interval_mean_price(
        self,
        timestamps: List[pd.Timestamp], 
        prices: List[float], 
        ref_time: pd.Timestamp, 
        timedelta_interval: TimedeltaInterval,
    ) -> float:
        df = pd.DataFrame({ "Timestamp": timestamps, "Price": prices })
        df = df[(
            (df["Timestamp"] >= ref_time + timedelta_interval.beg) &
            (df["Timestamp"] < ref_time + timedelta_interval.end)
        )]
        return df["Price"].mean()

    def transform(self, X: pd.DataFrame, y = None):
        # X = N x (datetime64, float)
        # find the latest timestamp
        X["Timestamp"] = X.apply(lambda row : pd.to_datetime(row["Date"]).max(), axis = "columns")

        # find rows that are between timedeltas with respect to the latest timestamp
        for col_name, timedelta_interval in self.timedelta_intervals.items():
            X[col_name] = X.apply(
                lambda row : self.interval_mean_price(
                    pd.to_datetime(row["Date"]),
                    row["Price"],
                    row["Timestamp"],
                    timedelta_interval,
                ),
                axis = "columns",
            )
            
        X = X.drop(columns=["Timestamp", "Date", "Price"])
        return X 


#for x in X.rolling(7*24*60*60, min_periods=7*24*60*60):
#    print(len(x))

timedelta_intervals = {
    "Current time": TimedeltaInterval(pd.Timedelta("-1s"), pd.Timedelta("1s")),
    "Last hour": TimedeltaInterval(pd.Timedelta("-1h"), pd.Timedelta("0s")),
    "Last 4 hours": TimedeltaInterval(pd.Timedelta("-4h"), pd.Timedelta("0s")),
    "Last day": TimedeltaInterval(pd.Timedelta("-1d"), pd.Timedelta("0s")),
    "Last week": TimedeltaInterval(pd.Timedelta("-7d"), pd.Timedelta("0s"))
}
pipeline = Pipeline(steps=[
    ("latest_time_interval_avgs", LatestTimeIntervalAverageTransformer(timedelta_intervals)),
    ("theil_sen_regressor", TheilSenRegressor())
])


N_samples = 1000
N_window = 8*24
train_df = df.copy(deep=True)

"""
train_df = pd.DataFrame(
    { 
        "Date": test_df["Date"][0], 
        "Price": test_df["Price"][0] 
    }
)
N_window = 4
"""

windows = train_df.rolling(N_window)
display(windows)
windows = [window for window in windows if len(window) >= N_window]
#for window in windows:
#    display(window)

Xyt = pd.DataFrame({
    "Date": [window["Date"].values[:-1] for window in windows],
    "Price": [window["Price"].values[:-1] for window in windows],
    "Next Price": [window["Price"].values[-1] for window in windows]
}).sample(N_samples)
display(Xyt.shape)
X = Xyt[["Date","Price"]]
y = Xyt["Next Price"]
display(X.dtypes)


"""
input_schema = Schema(
    [
        TensorSpec(np.dtype(np.float64), (-1, N_window), "Price"),
        TensorSpec(np.dtype(str), (-1, N_window), "Date"),
    ]
)
output_schema = Schema([TensorSpec(np.dtype(np.float64), (-1, 1))])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)
"""

with mlflow.start_run() as run:
    #autologgin päälle mlflowhun
    # without .copy(deep=True) => wrong signature
    model = pipeline.fit(X.copy(deep=True),y)
    #signature = infer_signature(X.head(2), model.predict(X.head(2).copy(deep=True)))
    #display(signature)
    model_info = mlflow.sklearn.log_model(
        sk_model=model, 
        artifact_path="model", # unity catalog
        #signature=signature,
        input_example=X.head(2).copy(deep=True),
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

X[["Price","Date"]].head(1).reset_index(drop=True).to_json(orient="records")

# COMMAND ----------

X[["Price","Date"]].head(1).reset_index(drop=True).to_json(orient="records")

# COMMAND ----------

X[["Price","Date"]].iloc[0].to_dict()

# COMMAND ----------

    signature = infer_signature(X.head(2), model.predict(X.head(2).copy(deep=True)))


# COMMAND ----------


