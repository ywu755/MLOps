from typing import NamedTuple, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric_temporal.signal import StaticGraphTemporalSignal


class GraphInfo(NamedTuple):
    """
    This class includes the relevant properties of a static graph.

    Args:
        edges: edges of a static graph
        num_nodes: number of nodes
        edge_weights: weights of the edges
        node_types: an array of types of nodes (e.g., generation, load)
    """
    edges: np.array
    num_nodes: int
    edge_weights: Optional[np.array] = None
    node_types: Optional[np.array] = None


class TemporalGNNDataset:
    """
    This class is used to build a custom StaticGraphTemporalSignal
    dataset based on a timeseries array, which can be used as the
    input for TemporalGNNModel.

    Args:
        X: a timeseries array (e.g., generation, load, additional
            features) with a shape of (time steps, number of nodes).
        graph_info: a GraphInfo class that includes the relevant
            properties of a static graph
        mean: an array of mean values that is used to normalize X.
            The shape of this array is (number of nodes,).
        std: an array of standard deviation values that is used to
            normalize X. The shape of this array is (number of nodes,).
    """

    def __init__(
        self,
        X: np.array,
        graph_info: GraphInfo,
        mean: Optional[np.array] = None,
        std: Optional[np.array] = None,
    ):
        self.X = X
        self.mean = mean
        self.std = std
        self.edges = graph_info.edges
        self.edge_weights = graph_info.edge_weights
        self.node_types = graph_info.node_types

    def _generate_task(
        self,
        num_timesteps_in: int,
        num_timesteps_out: int,
    ) -> Tuple[np.array, np.array]:
        """
        Generate features and targets based on the node feature(s)
        provided. The features array has the shape of (num_nodes,
        num_node_features, num_timesteps_in) and the targets array has
        the shape of (num_nodes, num_timesteps_out).
        If mean and std are provided, the features and targets array
        will be normalized.

        Args:
            num_timesteps_in: number of input timesteps
            num_timesteps_out: number of output timesteps

        Returns:
            Prepared tuple of arrays: (features, targets)
        """
        indices = [
            (i, i + (num_timesteps_in + num_timesteps_out))
            for i in range(
                self.X.shape[2] - (num_timesteps_in + num_timesteps_out) + 1
            )
        ]

        features, targets = [], []
        for i, j in indices:
            features.append(self.X[:, :, i:(i + num_timesteps_in)])
            targets.append(self.X[:, 0, (i + num_timesteps_in):j])
        features = np.array(features)
        targets = np.array(targets)

        if self.mean is None or self.std is None:
            return features, targets
        normalized_features = (
            features - self.mean.reshape(1, -1, 1, 1)
        ) / self.std.reshape(1, -1, 1, 1)

        normalized_targets = (
            targets - self.mean.reshape(1, -1, 1)
            ) / self.std.reshape(1, -1, 1)

        return normalized_features, normalized_targets

    def get_dataset(
        self,
        num_timesteps_in: int,
        num_timesteps_out: int,
    ) -> StaticGraphTemporalSignal:
        """
        Prepare the StaticGraphTemporalSignal class based on the
        num_timesteps_in and num_timesteps_out specified.

        Args:
            num_timesteps_in: number of input timesteps
            num_timesteps_out: number of output timesteps
        """
        features, targets = self._generate_task(
            num_timesteps_in, num_timesteps_out
        )
        return StaticGraphTemporalSignal(
            edge_index=self.edges,
            edge_weight=self.edge_weights,
            features=features,
            targets=targets,
        )


def convert_to_dataloader(
    dataset: StaticGraphTemporalSignal,
    batch_size: int,
    shuffle: Optional[bool] = True,
    drop_last: Optional[bool] = True,
    num_workers: Optional[int] = 0,
    pin_memory: Optional[bool] = False,
) -> DataLoader:
    """
    This function converts a StaticGraphTemporalSignal dataset
    class into a torch Dataloader.

    Args:
        dataset: a StaticGraphTemporalSignal class
    torch.utils.data.DataLoader Args:
        batch_size: number of samples to load per patch
        shuffle: whether to have data shuffled at every epoch
        drop_last: whether to drop the last incomplete batch
        num_workers: how many subprocesses to use for data loading
        pin_memory:  If True, the data loader will copy Tensors
            into device/CUDA pinned memory before returning them.
    """
    data_set_new = TensorDataset(
        torch.from_numpy(dataset.features).type(torch.FloatTensor),
        torch.from_numpy(dataset.targets).type(torch.FloatTensor),
    )
    return DataLoader(
        data_set_new,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
