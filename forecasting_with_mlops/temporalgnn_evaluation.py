
"""
This script includes customized metrics to evaluation balance and
load generation, and day-ahead temporalgnn model accuracy.
"""
from typing import Any, Tuple

import numpy as np
import torch
from torch import Tensor, tensor
from torchmetrics.metric import Metric


def _bal_squared_error_update(
    preds: Tensor, target: Tensor, node_types: np.ndarray
) -> Tuple[Tensor, int]:
    """
    Updates and returns variables required to compute balance and
    load differences.
    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        node_types: type of each node (generation or load)
    """
    load_node_types = node_types.reshape(1, -1, 1)
    gen_node_types = (1 - node_types).reshape(1, -1, 1)
    load = torch.sum(preds * load_node_types)
    gen = torch.sum(preds * gen_node_types)
    sum_squared_error = torch.square(load - gen)
    n_obs = node_types.shape[1]
    return sum_squared_error, n_obs


def _bal_mean_squared_error_compute(
    sum_squared_error: Tensor, n_obs: int, squared: bool = True
) -> Tensor:
    """
    Computes Mean Squared differences in gen and load balance.
    Args:
        sum_squared_error: Sum of square of differences over all buses
        n_obs: Number of buses
        squared: Returns sqrt value if set to False.
    """
    return (
        sum_squared_error / n_obs if squared else torch.sq(
            sum_squared_error / n_obs)
    )


class LoadGenerationBalance(Metric):
    """
    computer load and generation balance
    Args:
        squared: If True returns squared value, if False returns sqrt value.
        kwargs: Additional keyword arguments.
    """

    is_differentiable = True
    higher_is_better = False
    full_state_update = False
    sum_squared_error: Tensor
    total_examples: int

    def __init__(
        self,
        node_types: np.ndarray,
        squared: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.add_state(
            "sum_squared_error",
            default=tensor(0.0),
            dist_reduce_fx="sum")
        self.add_state(
            "total_examples",
            default=tensor(0),
            dist_reduce_fx="sum")
        self.squared = (squared,)
        self.node_types = node_types

    def update(
        self,
        preds: Tensor,
        target: Tensor,
    ) -> None:
        """
        Update state with predictions and targets.
        Args:
            preds: Predictions from model
            target: Ground truth values
            node_types: type of each node (generation or load)
        """
        sum_squared_error, n_obs = _bal_squared_error_update(
            preds, target, self.node_types
        )
        self.sum_squared_error += sum_squared_error
        self.total_examples += n_obs

    def compute(self) -> Tensor:
        """
        Computes mean squared differences between load and generation
        over state.
        """
        return _bal_mean_squared_error_compute(
            self.sum_squared_error, self.total_examples, squared=self.squared
        )


def total_loss(
    y: torch.Tensor, y_hat: torch.Tensor, loss_dict: dict, weight_dict: dict
) -> np.float64:
    """
    A combined loss function which allows a balance of both MSE and load
    gneration balance loss penalties.
    Args:
        y: Ground truth values
        y_hat: Predictions from model
        node_types: type of each node (generation or load)
        weight_parameter: a parameter to assign how much weight is given
            to each of the loss penalties.
    """
    sum_loss = 0.0
    for metric, func in loss_dict.items():
        sum_loss += (weight_dict[metric] * func(y, y_hat).detach().numpy())
        if weight_dict[metric] == 0.0:
            continue
        return sum_loss


def validate_weights(
    dict: dict
) -> dict:
    """
    A function that checks the weight of each parameter.
    The total weight must be <=1.
    Args:
        weight_dict: A weight dictionary.
    """
    weight_check = 0
    for metric, weight in dict.items():
        weight_check += weight
    if weight_check > 1:
        raise TypeError("total weight must be <= 1")
    return dict
