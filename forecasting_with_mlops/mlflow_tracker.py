
"""
This class contains a wrapper class for using MLFlow tracking functionality.
It initializes runs and performs different logging functionalities. There is
definitely more to build out / refactor in this.
"""
from typing import Any, Dict, List, Optional

import os

import mlflow
from mlflow import ActiveRun, pyfunc
from omegaconf import DictConfig
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "setup.py"],
    pythonpath=True,
)


class MLFlowTracker:
    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str,
        to_track: List[DictConfig],
        conda_environment: Optional[str] = None,
    ) -> None:
        """
        Initialized the MLFlow tracker class. Used for high level shorthand
        commands for interacting with MLFlow tracking functionality.

        Args:
            experiment_name: The name of the experiment to use. Creates new one
            if it doesn't already exist.
            tracking_uri: The location of the tracking registry, could be a
                file or sql lite location, both local or on the cloud.
            to_track: The list and locations of objects, interacts with hydra,
                to track.
            conda_environment: The environment file to track. Defaults to None.
        """

        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.to_track = to_track
        if not mlflow.is_tracking_uri_set():
            mlflow.set_tracking_uri(tracking_uri)
        env_path = os.environ.get("CONDA_PREFIX")
        if os.path.exists(env_path):
            os.system(f"conda env export > {conda_environment}")
        self.conda_environment = conda_environment
        self.experiment = mlflow.set_experiment(experiment_name)
        self.run_id = None

    def start_run(self) -> ActiveRun:
        """
        Starts a new run within the specified experiment.
        Returns:
            The active run.
        """
        run = mlflow.start_run(experiment_id=self.experiment.experiment_id)
        self.run_id = run.info.run_id
        return run

    def log_code(self, code_dir: str, output_dir: str) -> None:
        """
        Logs the .py files used in the model run. Both in the specified code
            dir as well as the root directory.
        Args:
            code_dir: The directory that contains the source code
            (i.e. src path)
            output_dir: The folder name where MLFlow should put this
            in tracker.
        """
        for file in os.listdir(code_dir):
            if (file != "__init__.py") and (file.split(".")[-1] == "py"):
                local_path = f"{code_dir}/{file}"
                mlflow.log_artifact(
                    local_path=local_path,
                    artifact_path=output_dir,
                )
        for file in os.listdir("./"):
            if (file != "__init__.py") and (file.split(".")[-1] == "py"):
                mlflow.log_artifact(
                    local_path=file,
                    artifact_path=output_dir,
                )

    def log_model(
            self,
            model: Any,
            model_logger: callable,
            output_dir: str
    ) -> None:
        """
        Tracks a pytorch model as a MLFlow artifact. Need to use a specific
        log_model function from MLFlow in order to register it in the MLFLow
        model registry.

        Args:
            model: The model object to track with MLFlow.
            model_logger: The function used to log the model.
            output_dir: The folder name where MLFlow should put this in
                tracker.
        """
        try:
            model_logger(
                model,
                artifact_path=output_dir,
                pickle_protocol=4,
                code_paths=[str(root)],
                conda_env=self.conda_environment,
            )
        except TypeError:
            pyfunc.log_model(model, artifact_path=output_dir)

    def log_and_end(
        self,
        model: Optional[Any] = None,
        model_logger: Optional[callable] = None,
        metrics: Optional[Dict[str, float]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Iterates through all of the objects that are desired to be tracked in
        the to_track dictionary, and calls each appropriate logging function
        according to the name of the object to track.
        Args:
            metrics: The model error metrics to track. Defaults to None.
            params: The model parameters to track. Defaults to None.
            model: The model object to track. Defaults to None.
            model_logger: The function used to log the model.
        Raises:
            ValueError: If an unknown object name is provided to track, raises
            an error.
        """

        for obj in self.to_track:
            if obj["name"] == "code":
                self.log_code(obj["location"], obj["output_location"])
            elif obj["name"] == "metrics":
                mlflow.log_metrics(metrics)
            elif obj["name"] == "params":
                mlflow.log_params(params)
            elif obj["name"] == "model":
                self.log_model(model, model_logger, obj["output_location"])
            else:
                raise ValueError(f"{obj['name']} not supported for tracking")
