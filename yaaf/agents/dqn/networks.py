from typing import Sequence, Union

import torch
from torch.nn import ModuleList, Linear
from torch.optim import Optimizer

from yaaf.models import TorchModel
from yaaf.models.feature_extraction import Conv2dFeatureExtractor, Conv2dSpec, LinearSpec, MLPFeatureExtractor, \
    FeatureExtractor


class DeepQNetwork(TorchModel):

    def __init__(self, feature_extractors: Sequence[FeatureExtractor],
                 num_actions: int,
                 learning_rate: float = 0.01,
                 optimizer: Union[str, Optimizer] = "adam",
                 cuda: bool = False):
        super(DeepQNetwork, self).__init__(learning_rate, optimizer, l2_penalty=0.0, loss="mse", dtype=torch.float32, cuda=cuda)
        self._feature_extractors = ModuleList(feature_extractors)
        self._q_values = Linear(self._feature_extractors[-1].num_features, num_actions)
        self.check_initialization()

    @property
    def input_shape(self):
        return self._feature_extractors[0].input_shape

    @property
    def feature_extractors(self):
        return self._feature_extractors

    def forward(self, X):
        for extractor in self._feature_extractors:
            X = extractor(X)
        Q = self._q_values(X)
        return Q

    def accuracy(self, X, y):
        """Correct prediction = greedy action"""
        greedy_actions = y.argmax(dim=-1)
        return super().accuracy(X, greedy_actions)


class FeedForwardDQN(DeepQNetwork):

    """
    DQN Architecture for feature-vector based observations
    """

    def __init__(self,
                 num_features: int,
                 num_actions: int,
                 mlp_layers: Sequence[LinearSpec] = ((64, "relu"), (64, "relu")),
                 learning_rate: float = 0.001,
                 optimizer: Union[str, Optimizer] = "adam",
                 cuda: bool = False):

        mlp_feature_extractor = MLPFeatureExtractor(num_inputs=num_features, layers=mlp_layers)
        super().__init__((mlp_feature_extractor,), num_actions, learning_rate, optimizer, cuda)


class DeepMindAtari2600DQN(DeepQNetwork):

    """
    DQN Architecture from https://www.nature.com/articles/nature14236
    """

    def __init__(self,
                 num_actions: int,
                 history_length: int = 4,
                 learning_rate: float = 0.0003,
                 optimizer="rmsprop",
                 cuda=True):

        conv2d_feature_extractor = \
            Conv2dFeatureExtractor(
                num_channels=history_length, width=84, height=84,
                layers=[
                    Conv2dSpec(num_kernels=32, kernel_size=8, stride=4, activation="relu"),
                    Conv2dSpec(num_kernels=64, kernel_size=4, stride=2, activation="relu"),
                    Conv2dSpec(num_kernels=64, kernel_size=3, stride=1, activation="relu")
                ]
            )

        mlp_feature_extractor = \
            MLPFeatureExtractor(
                num_inputs=conv2d_feature_extractor.num_features,
                layers=[
                    LinearSpec(512, "relu")
                ]
            )

        super().__init__((conv2d_feature_extractor, mlp_feature_extractor), num_actions, learning_rate, optimizer, cuda)