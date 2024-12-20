from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import A3TGCN2

from carbon_tracing.day_ahead_forecasting.temporalgnn_evaluation import (
    total_loss,
)


class TemporalGNNModel(pl.LightningModule):
    """
    This subclass utilizes pytorch.LightningModule to organize
    methods for training a temporal graph neural network model.
    The A3TGN2 model combines graph convolutional networks (GCN) and
    gated recurrent units (GRU) with an attention mechanism to weight
    historical signals. The details about the model is in "A3T-GCN:
    Attention Temporal Graph Convolutional Network for Traffic Forecasting."
    https://arxiv.org/abs/2006.11583.

    Args:
        node_features: number of input node features.
        out_channels: number of output node features.
        periods: number of time steps for prediction.
        batch_size: number of training example in one iteration.
        learning_rate: step size used for optimization.
        edge_index: edges of a static graph.
        edge_weight: weights of the edges.
        node_types: type of each node (generation or load)
    """

    def __init__(
        self,
        node_features: int,
        out_channels: int,
        periods: int,
        batch_size: int,
        learning_rate: float,
        edge_index: torch.Tensor,
        loss_dict: dict,
        weight_dict: dict,
        edge_weight: Optional[torch.Tensor] = None,
        node_types: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.a3tgn2 = A3TGCN2(
            in_channels=node_features,
            out_channels=out_channels,
            periods=periods,
            batch_size=batch_size,
        )
        self.linear = torch.nn.Linear(out_channels, periods)
        self.learning_rate = learning_rate
        self.edge_index = edge_index
        if edge_weight is not None:
            self.edge_weight = torch.abs(edge_weight)
        else:
            self.edge_weight = edge_weight
        self.node_types = node_types
        self.weight_dict = weight_dict
        self.loss_dict = loss_dict

    def forward(self, x):
        x = self.a3tgn2(x, self.edge_index, self.edge_weight)
        x = F.relu(x)
        x = self.linear(x)
        return x

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, train_batch, batch_idx) -> torch.Tensor:
        x, y = train_batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        total_loss_function = total_loss(
            y, y_hat, self.loss_dict, self.weight_dict)
        self.log("train_loss", loss)
        self.log("train_total_loss_function", total_loss_function)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        total_loss_function = total_loss(
            y, y_hat, self.loss_dict, self.weight_dict)
        self.log("val_loss", loss)
        self.log("val_total_loss_function", total_loss_function)
