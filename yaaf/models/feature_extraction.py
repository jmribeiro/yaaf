from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Sequence

import torch
from torch.nn import Module, ModuleList, Conv2d, Linear, Dropout, LSTM, GRU, RNN, Dropout2d
import matplotlib.pyplot as plt

from yaaf.models.utils import flatten, compute_output, activations

LinearSpec = namedtuple("LinearSpec", "units activation")
Conv2dSpec = namedtuple("Conv2dSpec", "num_kernels kernel_size stride activation")


class FeatureExtractor(Module, ABC):

    def __init__(self):
        super(FeatureExtractor, self).__init__()

    @property
    @abstractmethod
    def num_features(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def input_shape(self):
        raise NotImplementedError()


class ManualFeatureExtractor(FeatureExtractor, ABC):

    def __init__(self, input_shape: tuple, num_features: int):
        super().__init__()
        self._input_shape = input_shape
        self._num_features = num_features

    def forward(self, X):
        phi_X = self.feature_function(X)
        return phi_X

    @abstractmethod
    def feature_function(self, X):
        raise NotImplementedError()

    @property
    def num_features(self):
        return self._num_features

    @property
    def input_shape(self):
        return self._input_shape


class MLPFeatureExtractor(FeatureExtractor):

    def __init__(self, num_inputs: int, layers: Sequence[LinearSpec], dropout=0.0):

        super(MLPFeatureExtractor, self).__init__()
        self._num_inputs = num_inputs

        self._extractors = ModuleList()
        last_units = num_inputs

        for units, activation in layers:
            self._extractors.extend([
                Linear(last_units, units),
                activations[activation] if isinstance(activation, str) else activation,
                Dropout(dropout)
            ])
            last_units = units

        self._num_features = last_units

    def forward(self, X):
        for extractor in self._extractors:
            X = extractor(X)
        return X

    def features(self, X):
        with torch.no_grad():
            features = [extractor(X) for extractor in self._extractors if isinstance(extractor, Linear)]
        return features

    @property
    def num_features(self):
        return self._num_features

    @property
    def input_shape(self):
        return (self._num_inputs, )


class Conv2dFeatureExtractor(FeatureExtractor):

    def __init__(self, num_channels: int, width: int, height: int, layers: Sequence[Conv2dSpec], dropout=0.0):

        super(Conv2dFeatureExtractor, self).__init__()

        self._num_channels = num_channels
        self._width = width
        self._height = height

        self._extractors = ModuleList()
        last_kernels = num_channels

        for num_kernels, kernel_size, stride, activation in layers:
            self._extractors.extend([
                Conv2d(last_kernels, num_kernels, kernel_size, stride),
                activations[activation] if isinstance(activation, str) else activation,
                Dropout2d(dropout)
            ])
            last_kernels = num_kernels
            width, height = compute_output((width, height), kernel_size, stride)

        self._num_features = last_kernels * width * height

    def forward(self, X):
        for extractor in self._extractors:
            X = extractor(X)
        F = flatten(X)
        return F

    @property
    def num_features(self):
        return self._num_features

    @property
    def input_shape(self):
        return self._num_channels, self._width, self._height

    def feature_maps(self, X):
        feature_maps = []
        with torch.no_grad():
            for extractor in self._extractors:
                X = extractor(X)
                if isinstance(extractor, Conv2d):
                    w, h = X.shape[-2], X.shape[-1]
                    fmaps = X.view(-1, w, h).cpu()
                    [feature_maps.append(fmap) for fmap in fmaps]
        return feature_maps

    def plot_feature_maps(self, X):
        for feature_map in self.feature_maps(X):
            plt.imshow(feature_map)
            plt.show()

    @property
    def filters(self):
        filters = []
        for extractor in self._extractors:
            if isinstance(extractor, Conv2d):
                weights = extractor.weight.cpu().detach()
                w, h = weights.shape[-2], weights.shape[-1]
                weights = weights.view(-1, w, h).cpu()
                [filters.append(weight) for weight in weights]
        return filters

    def plot_filters(self):
        for filter in self.filters:
            plt.imshow(filter)
            plt.show()


class LSTMFeatureExtractor(FeatureExtractor):

    def __init__(self, num_inputs, num_layers, hidden_sizes, dropout, bidirectional):
        super().__init__()
        self._lstm = LSTM(num_inputs, hidden_sizes, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self._num_features = hidden_sizes * 2 if bidirectional else hidden_sizes

    def forward(self, X):
        _, (hidden, _) = self._lstm(X)
        F = torch.cat((hidden[-2], hidden[-1]), dim=1) if self._lstm.bidirectional else hidden[-1]
        return F

    @property
    def num_features(self):
        return self._num_features

    @property
    def input_shape(self):
        # TODO
        return 0
