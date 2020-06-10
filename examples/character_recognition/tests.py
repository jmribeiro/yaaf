from unittest import TestCase, main

import numpy as np
from sklearn import datasets

from examples.character_recognition.OCRDataset import OCRDataset
from examples.character_recognition.classifiers import FeedForwardNetwork, ConvNetForOCRDataset

from yaaf import rmdir


class SupervisedLearningTests(TestCase):

    def _test_mlp_ocr(self, binary, cuda, epochs):
        dataset = OCRDataset(binary=binary)
        model = FeedForwardNetwork(num_inputs=dataset.num_features, num_outputs=dataset.num_classes, learning_rate=0.001, layers=[(200, "relu")], dropout=0.3, cuda=cuda)
        model.update(dataset.X, dataset.y, epochs=epochs, batch_size=64, X_val=dataset.X_val, y_val=dataset.y_val)
        accuracy = model.accuracy(dataset.X_test, dataset.y_test)
        assert accuracy >= 0.85, f"Model only achieved {accuracy} accuracy."

    def _test_mlp_iris_overfit(self, cuda, epochs):
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        model = FeedForwardNetwork(num_inputs=X.shape[1], num_outputs=len(np.unique(y)), learning_rate=0.01, layers=[(64, "relu")], dropout=0.3, cuda=cuda)
        model.update(X, y, epochs=epochs, batch_size=32, X_val=X, y_val=y)
        accuracy = model.accuracy(X, y)
        assert accuracy >= 0.85, f"Model only achieved {accuracy} accuracy."

    def _test_cnn_ocr(self, binary, cuda, epochs):
        dataset = OCRDataset(image=True, binary=binary)
        model = ConvNetForOCRDataset(cuda)
        model.update(dataset.X, dataset.y, epochs=epochs, batch_size=64, X_val=dataset.X_val, y_val=dataset.y_val)
        accuracy = model.accuracy(dataset.X_test, dataset.y_test)
        assert accuracy >= 0.85, f"Model only achieved {accuracy} accuracy."

    def test_mlp_ocr_dataset_binary(self):
        self._test_mlp_ocr(binary=True, cuda=False, epochs=3)

    def test_mlp_ocr_dataset_binary_cuda(self):
        self._test_mlp_ocr(binary=True, cuda=True, epochs=3)

    def test_mlp_ocr_dataset(self):
        self._test_mlp_ocr(binary=False, cuda=False, epochs=20)

    def test_mlp_ocr_dataset_cuda(self):
        self._test_mlp_ocr(binary=False, cuda=True, epochs=20)

    def test_cnn_ocr_dataset_binary(self):
        self._test_cnn_ocr(binary=True, cuda=False, epochs=10)

    def test_cnn_ocr_dataset_binary_cuda(self):
        self._test_cnn_ocr(binary=True, cuda=True, epochs=10)

    def test_cnn_ocr_dataset(self):
        self._test_cnn_ocr(binary=False, cuda=False, epochs=30)

    def test_cnn_ocr_dataset_cuda(self):
        self._test_cnn_ocr(binary=False, cuda=True, epochs=30)

    def test_mlp_iris_dataset_overfit(self):
        self._test_mlp_iris_overfit(cuda=False, epochs=10)

    def test_mlp_iris_dataset_overfit_cuda(self):
        self._test_mlp_iris_overfit(cuda=True, epochs=10)


class PersistencyTests(TestCase):

    def _test_persistency(self, model_factory, X, y, epochs, acceptable_accuracy):

        """ Trains an arbitrary model, saves it, loads it, and tests if its still performing well. """

        # Make sure untrained model doesn't magically perform "well" on data
        untrained_accuracy = 1.0
        while untrained_accuracy > 0.55:
            model = model_factory()
            untrained_accuracy = model.accuracy(X, y)

        model.update(X, y, epochs=epochs, batch_size=32)
        pre_load_accuracy = model.accuracy(X, y)
        assert pre_load_accuracy >= acceptable_accuracy, f"Model only achieved {pre_load_accuracy} accuracy (required: {acceptable_accuracy})."
        model.save("tmp"); del model

        model = model_factory()
        model.load("tmp"); rmdir("tmp")
        post_load_accuracy = model.accuracy(X, y)
        assert post_load_accuracy >= pre_load_accuracy, f"Model only achieved {post_load_accuracy} accuracy. (required: {pre_load_accuracy})"

    def test_mlp_persistency(self):
        """ Trains a feedforward network, saves it, loads it, and tests if its still performing well. """
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        model_factory = lambda: FeedForwardNetwork(num_inputs=X.shape[1], num_outputs=len(np.unique(y)), learning_rate=0.01, layers=[(64, "relu")], dropout=0.3, cuda=True)
        self._test_persistency(model_factory, X, y, 10, 0.85)

    def test_cnn_persistency(self):
        """ Trains a convolutional network, saves it, loads it, and tests if its still performing well. """
        ocr = OCRDataset(image=True)
        model_factory = lambda: ConvNetForOCRDataset(cuda=True)
        self._test_persistency(model_factory, ocr.X, ocr.y, 30, 0.85)


if __name__ == '__main__':
    main()
