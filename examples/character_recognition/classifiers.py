import torch
from torch.nn import Linear
from yaaf.models import TorchModel
from yaaf.models.feature_extraction import MLPFeatureExtractor, Conv2dFeatureExtractor, Conv2dSpec
from yaaf.models.utils import activations


class FeedForwardNetwork(TorchModel):

    def __init__(self, num_inputs, num_outputs, layers,
                 learning_rate=0.001, optimizer="adam", l2_penalty=0.0, dropout=0.0,
                 loss="categorical cross entropy", output="linear", dtype=torch.long, cuda=False):

        super(FeedForwardNetwork, self).__init__(learning_rate, optimizer, l2_penalty, loss, dtype, cuda)
        self._mlp_feature_extractor = MLPFeatureExtractor(num_inputs, layers, dropout)
        self._output = Linear(self._mlp_feature_extractor.num_features, num_outputs)
        self._output_activation = activations[output] if output != "linear" else lambda x: x

    def forward(self, X):
        F = self._mlp_feature_extractor(X)
        Z = self._output_activation(self._output(F))
        return Z

    def features(self, X):
        return self._mlp_feature_extractor.features(X)


class LogisticRegression(TorchModel):

    def __init__(self, num_inputs, num_outputs, learning_rate, optimizer, l2_penalty, loss, dtype, cuda):
        super().__init__(learning_rate, optimizer, l2_penalty, loss, dtype, cuda)
        self._linear = Linear(num_inputs, num_outputs)

    def forward(self, X):
        # Z = W.X + b
        Z = self._linear(X)
        return Z


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


class ConvNetForOCRDataset(ConvolutionalNeuralNetwork):

    def __init__(self, cuda):
        # Surely not the best possible architecture, but gets the job done!
        conv1 = Conv2dSpec(num_kernels=16, kernel_size=5, stride=1, activation="relu")
        conv2 = Conv2dSpec(num_kernels=32, kernel_size=2, stride=1, activation="relu")
        super(ConvNetForOCRDataset, self).__init__(
            num_channels=1, width=8, height=16,
            num_outputs=26, convolutional_layers=[conv1, conv2], fully_connected_layers=[],
            learning_rate=0.001, dropout=0.3, cuda=cuda)
