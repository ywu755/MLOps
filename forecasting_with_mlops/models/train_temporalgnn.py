
"""
The purpose of this script is to train a temporal GNN model for
forecasting day-ahead load and generation in the ComEd test site.
"""
from typing import Any, Dict, Optional, Tuple

import os
from argparse import ArgumentParser

import mlflow
import networkx as nx
import numpy as np
from omegaconf import OmegaConf
import optuna
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import torch
import torchmetrics
from torch_geometric_temporal.signal import temporal_signal_split
from urllib.parse import urlparse

from forecasting.temporalgnn_data import (
    GraphInfo,
    TemporalGNNDataset,
    convert_to_dataloader,
)
from forecasting.temporalgnn_evaluation import (
    LoadGenerationBalance,
    validate_weights,
)
from forecasting.temporalgnn_model import (
    TemporalGNNModel,
)

from mlflow_tracker import (
    MLFlowTracker,
)

INPUT_NAME = "MW"
TIMESTAMP = "Timestamp"
BUS = "Bus"
LOAD_GEN_LIST = ["load", "gen"]

TB_LOG_PATH = ""
STUDY_PATH = None
STUDY_NAME = "optuna-mlflow-study"
R_THRES = 0
TRAIN_RATIO = 0.5
VAL_RATIO = 0.25
TIMESTEPS = 8760
NUM_TIMESTEPS_IN = 720
NUM_TIMESTEPS_OUT = 24
BATCH_SIZE = 32
NUM_WORKERS = 8
WEIGHT_PARAMETER = 0.5
EXPERIMENT_NAME = "gnn-optuna-mlflow-test"
WEIGHT_DICT = {
    "MSE": 0.5, "MAE": 0.0, "MAPE": 0.0, "load_generation_bal": 0.5
}
MSE_CLASS = torchmetrics.MeanSquaredError()
MAE_CLASS = torchmetrics.MeanAbsoluteError()
MAPE_CLASS = torchmetrics.MeanAbsolutePercentageError()
MODEL_CLASS = mlflow.pytorch.log_model

LOSS_DICT = {
    "MSE": MSE_CLASS, 
    "MAE": MAE_CLASS,
    "MAPE": MAPE_CLASS,
}

# ComEd data
CLEANSED_OUT_DATA = "gs://kv_ds_carbon_comed/out_data"
CLEANSED_PATH = "gs://kv_ds_carbon_comed/cleansed"
CLEANSED_FILE = "cleansed"


def _process_cleansed_file(
    cleansed_data: pd.DataFrame
) -> Tuple[Optional[np.array], Optional[np.array]]:
    """
    Prepare cleansed load and generation timeseries data used as the input for
    TemporalGNNDataset class.

    Args:
       cleansed_data: data from the cleanse process.
    """
    cleansed_MW = (
        cleansed_data[cleansed_data.BusType.isin(LOAD_GEN_LIST)][[TIMESTAMP, BUS, INPUT_NAME]]
        .pivot(index=TIMESTAMP, columns=BUS)[INPUT_NAME])

    cleansed_MW = (
        cleansed_MW.loc[:, (cleansed_MW.sum(axis=0) != 0)]
        .fillna(0)
    )
    
    # prepare data for Bus_Type
    cleansed_bus_types = (
        cleansed_data[cleansed_data.Bus.isin(cleansed_MW.columns)]
        [[TIMESTAMP, BUS, "BusType"]]
        .query('BusType in @LOAD_GEN_LIST')
        .assign(Type=lambda x: np.where(x.BusType == "load", 1, np.where(x.BusType == "gen", 0, 99)))
        .drop("BusType", axis=1)
        .drop_duplicates(subset=["Bus", "Type"])
        [["Bus", "Type"]])
    
    cleansed_bus_types["temp_index"] = 0

    cleansed_bus_types = cleansed_bus_types.pivot(
        index="temp_index", columns=BUS, values="Type"
    )
    return cleansed_MW.to_numpy(), cleansed_bus_types.to_numpy()


def build_corr_graph(
    ts_array: np.array,
    r_thres: float,
) -> nx.graph:
    """
    Build a graph based on the correlation between a set of timeseries
    data.

    Args:
        ts_array: an array of timeseries data.
        r_thres: a threshold of correlation coefficient, the graph would
            only keep edges with r >= r_thres.
    """
    r_df = pd.DataFrame(ts_array).corr().stack().reset_index()
    r_df.columns = ["node1", "node2", "r"]

    return nx.from_pandas_edgelist(
        r_df.loc[(abs(r_df["r"]) >= r_thres) & (r_df["node1"] != r_df["node2"])],
        "node1",
        "node2",
        edge_attr=True,
    )


def _training_data_prep():
    ts_array, bus_types_array = _process_cleansed_file(
        pd.read_csv(f"{CLEANSED_PATH}/{CLEANSED_FILE}.csv")
    )

    # Prepare GraphInfo
    g_corr = build_corr_graph(ts_array, R_THRES)
    graph_info = GraphInfo(
        edges=np.array([[edge_from, edge_to] for edge_from, edge_to in g_corr.edges]).T,
        edge_weights=np.array([g_corr.edges[edge]["r"] for edge in g_corr.edges]),
        num_nodes=len(g_corr.nodes),
        node_types=bus_types_array,
    )
    # Prepare training & validation data
    train_timestep = int(TIMESTEPS * TRAIN_RATIO)
    loader = TemporalGNNDataset(
        X=ts_array.T.reshape((graph_info.num_nodes, 1, TIMESTEPS)),
        graph_info=graph_info,
        mean=np.mean(ts_array[0:train_timestep, :], axis=0),
        std=np.std(ts_array[0:train_timestep, :], axis=0),
    )
    dataset = loader.get_dataset(
        num_timesteps_in=NUM_TIMESTEPS_IN, num_timesteps_out=NUM_TIMESTEPS_OUT
    )
    _dataset, test_dataset = temporal_signal_split(
        dataset, train_ratio=TRAIN_RATIO + VAL_RATIO
    )
    train_dataset, val_dataset = temporal_signal_split(
        _dataset, train_ratio=TRAIN_RATIO
    )
    return (
        graph_info,
        convert_to_dataloader(
            train_dataset,
            BATCH_SIZE,
            drop_last=True,
            num_workers=NUM_WORKERS,
        ),
        convert_to_dataloader(
            val_dataset,
            BATCH_SIZE,
            shuffle=False,
            drop_last=True,
            num_workers=NUM_WORKERS,
        ),
    )


# Obtain hyperparameters for this trial
def suggest_hyperparameters(trial):
    out_channels = trial.suggest_int("out_channels", 1, 64, log=True)
    # Obtain the learning rate
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    return out_channels, learning_rate


def objective(
    trial: optuna.trial.Trial,
    experiment_name: str,
    max_epochs: int,
    patience: int, 
) -> float:

    tracker = MLFlowTracker(
        experiment_name=experiment_name,
        tracking_uri=conf["tracking_uri"],
        to_track=conf["to_track"],
    )

    early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=1e-4,
            patience=patience,
            verbose=False,
            mode="min",
        )

    trainer = pl.Trainer(
        callbacks=[early_stop_callback], max_epochs=max_epochs
    )

    # # Start a new mlflow run
    with tracker.start_run() as active_run:
        print(f"Starting run {active_run.info.run_id} and trial {trial.number}")

        # Parse the active mlflow run's artifact_uri
        # and convert it into a system path
        parsed_uri = urlparse(active_run.info.artifact_uri)
        artifact_path = os.path.abspath(
            os.path.join(parsed_uri.netloc, parsed_uri.path)
        )
        print(f"Artifact path for this run: {artifact_path}")

        # Create train and validation loaders
        graph_info, train_loader, val_loader = _training_data_prep()
        static_edge_index = torch.LongTensor(graph_info.edges)
        static_edge_weight = torch.Tensor(graph_info.edge_weights)
        node_types = graph_info.node_types

        periods = NUM_TIMESTEPS_OUT
        edge_index = static_edge_index

        # Get hyperparameter suggestions created by optuna
        out_channels, learning_rate = suggest_hyperparameters(trial)
        edge_weight = static_edge_weight

        weight_dict = validate_weights(WEIGHT_DICT)

        loss_dict: Dict[str, Any] = LOSS_DICT.copy()
        loss_dict["load_generation_bal"] = LoadGenerationBalance(node_types)

        model = TemporalGNNModel(
            node_features=1,
            out_channels=out_channels,
            periods=periods,
            batch_size=BATCH_SIZE,
            edge_index=edge_index,
            edge_weight=edge_weight,
            learning_rate=learning_rate,
            loss_dict=loss_dict,
            weight_dict=weight_dict,
        )

        trainer.fit(model, train_loader, val_loader)

        metrics_dict = {
            key: value.item()
            for key, value
            in trainer.callback_metrics.items()
        }
        
        metrics_dict.update(weight_dict)
        
        tracker.log_and_end(
            model=trainer.model,
            model_logger=MODEL_CLASS,
            metrics=metrics_dict,
            params=trial.params,
        )
    return trainer.callback_metrics["val_total_loss_function"].item()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_yaml_path",
        type=str,
        metavar="",
        help="path to yaml configuration file. Example of yaml doc located at"
        "carbon-tracing/reference/tracker.yaml",
    )
    parser.add_argument(
        "-n",
        "--n_trials",
        type=int,
        default=1,
        help="Number of trials.",
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=int,
        default=6000,
        help="Stop study after the given number of seconds.",
    )
    parser.add_argument(
        "-e",
        "--max_epochs",
        type=int,
        default=20,
        help="Max number of epochs to run model for.",
    )
    parser.add_argument(
        "-p",
        "--patience",
        type=int,
        default=10,
        help="How long to wait for improvement in patience metric.",
    )
    args = parser.parse_args()

    # folder to load config file
    config_path = args.config_yaml_path
    # Function to load yaml configuration file from a path
    conf = OmegaConf.load(config_path)

    # Create the optuna study which shares the experiment name
    study = optuna.create_study(
        study_name=EXPERIMENT_NAME,
        direction="minimize",
    )
    study.optimize(
        lambda trial: objective(
            trial,
            experiment_name=EXPERIMENT_NAME,
            max_epochs=args.max_epochs,
            patience=args.patience,
        ),
        n_trials=args.n_trials, 
        timeout=args.timeout,
    )
