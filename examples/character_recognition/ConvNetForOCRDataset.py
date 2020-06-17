from yaaf.models.ConvolutionalNetwork import ConvolutionalNeuralNetwork
from yaaf.models.feature_extraction import Conv2dSpec


class ConvNetForOCRDataset(ConvolutionalNeuralNetwork):

    def __init__(self, cuda):
        # Surely not the best possible architecture, but gets the job done!
        conv1 = Conv2dSpec(num_kernels=16, kernel_size=5, stride=1, activation="relu")
        conv2 = Conv2dSpec(num_kernels=32, kernel_size=2, stride=1, activation="relu")
        super(ConvNetForOCRDataset, self).__init__(
            num_channels=1, width=8, height=16,
            num_outputs=26, convolutional_layers=[conv1, conv2], fully_connected_layers=[],
            learning_rate=0.001, dropout=0.3, cuda=cuda)
