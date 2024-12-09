"""
Train a Neural Prophet model for forecasting day-ahead load and generation in
the National Grid test site.
"""
import os
from typing import Literal, Optional, Tuple, Union

import mlflow
from argparse import ArgumentParser

from omegaconf import OmegaConf
from neuralprophet import NeuralProphet
import optuna
import pandas as pd

from forecasting_with_mlops.mlflow_tracker import (
    MLFlowTracker,
)
import data.constants as c

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_INPUT_DIR = f"{BASE_DIR}/input_data"
MODEL_OUTPUT_DIR = f"{BASE_DIR}/out_data"
CLEANSED_PATH = f"{MODEL_OUTPUT_DIR}/cleansed"
CLEANSED_OUT_DATA = f"{MODEL_OUTPUT_DIR}/cleansed"
CLEANSED_FILE = "Cleansed_MW"
LOAD_GEN_LIST = ["load", "gen"]
HOURS = 24
DS = "ds"
Y = "y"
METRIC_LIST = ["MAE", "RMSE", "MAE_val", "RMSE_val"]

PARAM_BOUNDS = {
    "n_lags": [0, 24],
    "learning_rate": [0.001, 0.01],
    "num_hidden_layers": [1, 2],
    "d_hidden": [1, 24],
    "daily_seasonality": [True, False],
    "weekly_seasonality": [True, False],
    "yearly_seasonality": [True, False],
    "growth": ["off", "linear"],
    "country_holidays": [None, "US"],
    "ar_reg": [0.0001, 1],
}
EXPERIMENT_NAME = "neuralprophet-optuna-mlflow-test"
MODEL_CLASS = mlflow.pytorch.log_model


def train_neuralprophet_model(
    profile: pd.DataFrame,
    input_name: str,
    timestamp: str,
    daily_seasonality: Union[int, Literal["auto", True, False]],
    weekly_seasonality: Union[bool, Literal["auto", True, False]],
    yearly_seasonality: Union[bool, Literal["auto", True, False]],
    n_lags: Optional[int] = None,
    growth: Optional[str] = None,
    learning_rate: Optional[float] = None,
    num_hidden_layers: Optional[int] = None,
    d_hidden: Optional[int] = None,
    country_holidays: Optional[str] = None,
    ar_reg: Optional[float] = None,
    epochs: Optional[int] = None,
) -> Tuple[
        Optional[pd.DataFrame], Optional[pd.Series], Optional[NeuralProphet]
        ]:
    """
    Train a NeuralProphet model on historical data and product multi hours
    forecasts.
    Args:
        profile: an array of timeseries data for a bus.
        input_name: a variable name of the model input.
        timestamp: Timestamp
        daily_seasonality: seasonal components to be modelled.
        weekly_seasonality: seasonal components to be modelled.
        yearly_seasonality: seasonal components to be modelled.
        n_lags: A number of lags determines how far into the past
            the auto-regressive dependencies should be considered.
            If n_lags > 0, the AR-Net is enabled.
        growth: trend growth type
        learning_rate: learning rate
        num_hidden_layers: a number that defines the number of hidden layers
            of the FFNNs used in the overall model.
        d_hidden: the number of units in the hidden layers.
        country_holidays: a list of country specific holidays.
        ar_reg: how much sparsity to induce in the AR-coefficients.
        epochs: Number of epochs (complete iterations over dataset) to
            train model.
    """

    prophet_data = (
        profile.reset_index()[[timestamp, input_name]]
        .rename(columns={timestamp: "ds", input_name: "y"})
        .copy()
    )

    # Define training and testing datasets
    df_train = prophet_data.drop(prophet_data.index[-HOURS:])
    df_test = prophet_data[-HOURS:]

    neural_forecaster = NeuralProphet(
        daily_seasonality=daily_seasonality,
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality,
        n_lags=n_lags,
        growth=growth,
        num_hidden_layers=num_hidden_layers,
        learning_rate=learning_rate,
        d_hidden=d_hidden,
        ar_reg=ar_reg,
        epochs=epochs,
    )
    if country_holidays is not None:
        neural_forecaster.add_country_holidays(country_name=country_holidays)

    # fit the model
    neuralprophet_model_metrics = neural_forecaster.fit(
        df_train,
        freq="H",
        validation_df=df_test
    ).tail(1)

    # make a prediction
    forecast = neural_forecaster.predict(df_test)
    return (
        forecast[[DS, Y]].set_index(DS),
        neuralprophet_model_metrics,
        neural_forecaster,
    )


def objective(
    trial: optuna.trial.Trial,
    experiment_name: str,
    max_epochs: int,
) -> float:
    tracker = MLFlowTracker(
        experiment_name=experiment_name,
        tracking_uri=conf["tracking_uri"],
        to_track=conf["to_track"],
    )

    buses = pd.read_csv(f"{CLEANSED_PATH}/{CLEANSED_FILE}.csv")
    buses = buses[buses.BusType.isin(LOAD_GEN_LIST)]
    bus_id = buses.Bus.unique()
    output_data = pd.DataFrame()
    output_metrics = pd.DataFrame()

    with mlflow.start_run():
        model_parameters = {}
        for param in ["n_lags", "num_hidden_layers", "d_hidden"]:
            model_parameters[param] = trial.suggest_int(
                param, PARAM_BOUNDS[param][0], PARAM_BOUNDS[param][1]
            )
        for param in ["learning_rate", "ar_reg"]:
            model_parameters[param] = trial.suggest_float(
                param, PARAM_BOUNDS[param][0], PARAM_BOUNDS[param][1], log=True
            )
        for param in [
            "daily_seasonality",
            "weekly_seasonality",
            "yearly_seasonality",
            "growth",
            "country_holidays",
        ]:
            model_parameters[param] = trial.suggest_categorical(
                param, PARAM_BOUNDS[param]
            )

        # iterate over Buses
        for bus in bus_id.tolist():
            bus_profile = buses[buses.Bus == bus][[c.TIMESTAMP, c.INPUT_NAME]]
            (
                model_results,
                neuralprophet_model_metrics,
                neural_forecaster,
            ) = train_neuralprophet_model(
                bus_profile,
                input_name=c.INPUT_NAME,
                timestamp=c.TIMESTAMP,
                epochs=max_epochs,
                **model_parameters,
            )
            model_results["Bus"] = bus
            neuralprophet_model_metrics["Bus"] = bus
            output_metrics = output_metrics.append(neuralprophet_model_metrics)
            output_data = output_data.append(model_results)
            output_metrics.to_csv(
                f"{CLEANSED_OUT_DATA}/neuralprophet_metrics.csv"
            )

            metrics = output_metrics[METRIC_LIST].mean()
            metrics_dict = {keys: metrics[keys] for keys in METRIC_LIST}

            tracker.log_and_end(
                model=neural_forecaster.model,
                model_logger=MODEL_CLASS,
                metrics=metrics_dict,
                params=trial.params,
            )
        return output_metrics["RMSE"].mean()


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_yaml_path",
        type=str,
        metavar="",
        help="path to yaml configuration file. Example of yaml doc located at"
        "reference/prophet_model_config.yaml",
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
    args = parser.parse_args()
    config_path = args.config_yaml_path

    conf = OmegaConf.load(config_path)

    # Create the optuna study which shares the experiment name
    study = optuna.create_study(
        study_name=EXPERIMENT_NAME,
        direction="minimize"
    )
    study.optimize(
        lambda trial: objective(
            trial,
            experiment_name=EXPERIMENT_NAME,
            max_epochs=args.max_epochs,
        ),
        n_trials=args.n_trials,
        timeout=args.timeout,
    )
