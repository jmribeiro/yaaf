import torch
from torch.nn import Linear
from yaaf.models import TorchModel
from yaaf.models.feature_extraction import MLPFeatureExtractor, Conv2dFeatureExtractor
from yaaf.models.utils import activations


class ConvolutionalNeuralNetwork(TorchModel):

    def __init__(self, num_channels, width, height, num_outputs,
                 convolutional_layers, fully_connected_layers,
                 learning_rate, optimizer="adam", l2_penalty=0.0, dropout=0.0, loss="categorical cross entropy", output="linear", dtype=torch.long, cuda=False):

        super(ConvolutionalNeuralNetwork, self).__init__(learning_rate, optimizer, l2_penalty, loss, dtype, cuda)

        self._convolutional_feature_extractor = Conv2dFeatureExtractor(num_channels, width, height, convolutional_layers, dropout)
        self._mlp_feature_extractor = MLPFeatureExtractor(self._convolutional_feature_extractor.num_features, fully_connected_layers, dropout)
        self._output_layer = Linear(self._mlp_feature_extractor.num_features, num_outputs)
        self._output_activation = activations[output] if output != "linear" else lambda x: x

    def forward(self, X):
        F = self._convolutional_feature_extractor(X)
        F = self._mlp_feature_extractor(F)
        return self._output_activation(self._output_layer(F))

    def feature_maps(self, X):
        return self._convolutional_feature_extractor.feature_maps(X)

    def plot_feature_maps(self, X):
        self._convolutional_feature_extractor.plot_feature_maps(X)

    @property
    def filters(self):
        return self._convolutional_feature_extractor.filters

    def plot_filters(self):
        return self._convolutional_feature_extractor.plot_filters()