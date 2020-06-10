from logging import warning

import torch
from sklearn.metrics import accuracy_score, f1_score

from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from yaaf.models import utils
from yaaf.models.utils import TorchDataset, RawDataset, assure_tensor
from yaaf import mkdir


class TorchModel(Module):

    def __init__(self, learning_rate, optimizer, l2_penalty, loss, dtype, cuda):

        super(TorchModel, self).__init__()

        self._optimization_specs = dict(optimizer=optimizer, loss=loss)
        self._optimizer_initialized = False
        self._deployed_to_device = False

        self._learning_rate = learning_rate
        self._l2_penalty = l2_penalty
        self._output_dtype = dtype
        self._cuda = cuda

    # ######## #
    # Training #
    # ######## #

    def fit(self, X, y, epochs, batch_size, shuffle_dataset=True):

        """
        Yields the avg batch loss
        X:
        y:
        epochs:
        batch_size:
        shuffle_dataset:
        """

        self._check_device()
        self._check_optimizer_initialization()

        train_losses = []
        train_mean_losses = []

        batch_size = min(len(X), batch_size)
        dataloader = DataLoader(RawDataset(X, y, self._output_dtype), batch_size=batch_size, shuffle=shuffle_dataset)

        for _ in range(epochs):
            epoch_losses = self._fit_epoch(dataloader)
            train_losses.extend(epoch_losses)
            mean_epoch_loss = torch.tensor(train_losses).mean().item()
            train_mean_losses.append(mean_epoch_loss)
            yield train_mean_losses[-1]

    def _fit_epoch(self, dataloader):
        if not self.training: self.train()
        losses = [self._fit_batch(X_batch, y_batch) for X_batch, y_batch in dataloader]
        return losses

    def _fit_batch(self, X, y):
        # TODO since this method is only used through fit, move full data conversion to fit and gain a few calls ;)
        X = self.data_to_model(X, torch.float32)
        y = self.data_to_model(y, self._output_dtype)
        self._optimizer.zero_grad()
        y_hat = self(X)
        loss = self._loss(y_hat, y)
        loss.backward()
        self._optimizer.step()
        return loss.detach().item()

    # ########## #
    # Evaluation #
    # ########## #

    def predict(self, X):
        if self.training: self.eval()
        self._check_device()
        with torch.no_grad():
            X = self.data_to_model(X, torch.float32)
            Z = self(X).detach().cpu()
            return Z

    def classify(self, X):
        scores = self.predict(X)
        predicted_labels = scores.argmax(dim=-1)
        return predicted_labels

    # ########### #
    # Persistence #
    # ########### #

    def save(self, directory):
        """ Saves model and optimizer's state to a directory. """
        mkdir(directory)
        self.check_initialization()
        torch.save(self.state_dict(), f"{directory}/model.pt")
        torch.save(self._optimizer.state_dict(), f"{directory}/optimizer.pt")

    def load(self, directory):
        """ Loads model and optimizer's state from a directory. """
        self.check_initialization()
        model_state_dict = torch.load(f"{directory}/model.pt")
        optimizer_state_dict = torch.load(f"{directory}/optimizer.pt")
        self.load_state_dict(model_state_dict)
        self._optimizer.load_state_dict(optimizer_state_dict)

    # ######### #
    # Auxiliary #
    # ######### #

    def check_initialization(self):
        self._check_device()
        self._check_optimizer_initialization()

    def _check_device(self):
        if not self._deployed_to_device:
            self._deploy_to_device()
            self._deployed_to_device = True

    def _check_optimizer_initialization(self):
        if not self._optimizer_initialized:
            self.reset_optimizer_state()
            self._optimizer_initialized = True

    def _deploy_to_device(self):

        self._device = torch.device("cuda:0" if torch.cuda.is_available() and self._cuda else "cpu")

        if self._cuda and not torch.cuda.is_available():
            warning(f"Unable to deploy model into device {self._device} (cuda not available)."
                  f" Using device {self._device}.")
        else:
            try:
                self.to(self._device)
            except RuntimeError as e:
                self._device = torch.device("cpu")
                warning(f"Unable to deploy model into device {self._device} ({e}). "
                      f"Using device {self._device}.")
                self.to(self._device)

    def reset_optimizer_state(self):

        if hasattr(self, "_optimizer"):
            self._optimizer.zero_grad()
            del self._optimizer

        if hasattr(self, "_loss"):
            self._loss.zero_grad()
            del self._loss

        self._optimizer = utils.optimizers[self._optimization_specs["optimizer"]](params=self.parameters(), lr=self._learning_rate, weight_decay=self._l2_penalty) if isinstance(self._optimization_specs["optimizer"], str) else self._optimization_specs["optimizer"]
        self._loss = utils.losses[self._optimization_specs["loss"]] if isinstance(self._optimization_specs["loss"], str) else self._optimization_specs["loss"]

    def data_to_model(self, data, dtype):
        """ Converts data into Tensor of respective dtype. """
        data = assure_tensor(data, dtype)
        self._check_device()
        if data.device != self._device:
            data = data.to(self._device)
        return data

    # ########## #
    # Deprecated #
    # ########## #

    def accuracy(self, X, y):
        """ Use sklearn instead """
        y_hat = self.classify(X)
        accuracy = accuracy_score(y, y_hat)
        return accuracy

    def f1(self, X, y):
        """ Use sklearn instead """
        y_hat = self.classify(X)
        accuracy = f1_score(y, y_hat)
        return accuracy

    def update(self, X, y, epochs, batch_size,
               validation_split=0.0, X_val=None, y_val=None,
               observers=None, verbose=False):

        """ Use fit instead """

        self._check_device()
        self._check_optimizer_initialization()

        if not self.training: self.train()

        dataset = TorchDataset(X, y, self._output_dtype, validation_split, X_val, y_val)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        observers = observers or []
        train_losses = []
        train_mean_losses = []
        train_accuracies = []
        validation_accuracies = []

        for epoch in range(1, epochs + 1):
            if verbose: print(f"\nTraining epoch {epoch}/{epochs}", flush=True)
            iterator = tqdm(dataloader) if verbose else dataloader
            train_losses.extend([self._fit_batch(X_batch, y_batch) for X_batch, y_batch in iterator])
            train_mean_losses.append(torch.tensor(train_losses).mean().item())
            train_accuracies.append(self.accuracy(dataset.X, dataset.y))
            validation_accuracies.append(
                train_accuracies[-1] if dataset.no_validation_data else self.accuracy(dataset.X_val, dataset.y_val))
            metrics = {
                "training loss": train_mean_losses[-1],
                "training accuracy": train_accuracies[-1],
                "validation accuracy": validation_accuracies[-1]
            }
            [observer(epoch, **metrics) for observer in observers]
            if verbose:
                for metric_name in metrics:
                    print(f"{metric_name}: {round(metrics[metric_name], 4)}", flush=True)

        return train_mean_losses, train_accuracies, validation_accuracies
