# Azure Databrick ML demo

## Nordpool sklearn predictive API

Creates an sklearn pipeline that converts two weeks of hourly price data to past weeks average, past days average, past 4h average, past hour's price, and current hour's price, and then fits Theil Sun Regressor to a random sample of Nordpool's history data. The pipeline is then saved as MLFlow model to the model registry that can then be deployed as an API using a Model serving endpoint.
