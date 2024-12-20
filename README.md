# Forecasting with MLOps
This repo is a tool for structuring, tracking data science and machine learning projects.

The mlflow tracker contains a wrapper class for using `MLFlow` tracking functionality.
It initializes runs and performs different logging functionalities.


## Short Term Forecasting
This component includes methodology used for forecasting the short term load (day-ahead) and generation in an hourly resolution.

`NeuralProphet` is built on PyTorch and combines Neural Networks and traditional time-series algorithms, inspired by Facebook Prophet and AR-Net.

`temporalgnn_data.py` includes a Dataset class that can process load and generation timeseries for the purpose of training a graph-based forecasting model using torch_geometric_temporal.

`temporalgnn_model.py` includes the pytorch_lightning implementation of the A3TGN2 model (which combines graph convolutional networks (GCN) and gated recurrent units (GRU)) for training, validation, and prediction.
