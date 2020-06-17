import torch
from torch.nn import Linear
from yaaf.models import TorchModel
from yaaf.models.feature_extraction import MLPFeatureExtractor
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
