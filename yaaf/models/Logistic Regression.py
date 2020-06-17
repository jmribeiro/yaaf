from torch.nn import Linear

from yaaf.models import TorchModel


class LogisticRegression(TorchModel):

    def __init__(self, num_inputs, num_outputs, learning_rate, optimizer, l2_penalty, loss, dtype, cuda):
        super().__init__(learning_rate, optimizer, l2_penalty, loss, dtype, cuda)
        self._linear = Linear(num_inputs, num_outputs)

    def forward(self, X):
        Z = self._linear(X)
        return Z