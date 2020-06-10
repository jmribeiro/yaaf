import torch
from torch.nn import ReLU, Tanh, CrossEntropyLoss, Sigmoid, SELU, MSELoss, L1Loss, SmoothL1Loss, NLLLoss, BCELoss
from torch.optim import Adam, SGD, RMSprop, Adagrad
from torch.utils.data.dataset import Dataset


activations = {
    "relu": ReLU(),
    "tanh":  Tanh(),
    "sigmoid": Sigmoid(),
    "selu": SELU(),
}


optimizers = {
    "adam": Adam,
    "sgd": SGD,
    "rmsprop": RMSprop,
    "adagrad": Adagrad,
}

losses = {
    "negative log likelihood": NLLLoss(),
    "nll": NLLLoss(),
    "binary cross entropy": BCELoss(),
    "bce": BCELoss(),
    "categorical cross entropy": CrossEntropyLoss(),
    "cce": CrossEntropyLoss(),
    "mean squared error": MSELoss(),
    "mse": MSELoss(),
    "mean absolute error": L1Loss(),
    "mae": L1Loss(),
    "huber loss": SmoothL1Loss(),
}


class RawDataset(Dataset):

    """ Simple Wrapper for torch dataset (used by classes such as TorchModel and Dataloader) """

    def __init__(self, X, y, output_dtype):
        self.X, self.y = assure_tensor(X, torch.float32), assure_tensor(y, output_dtype)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def unflatten(data, dims):
    """ Given a batch of flattened 1-D tensors, reshapes them into specified N-Dims. """
    B = data.size(0)
    return data.view(B, *dims)


def flatten(data):
    """ Given a batch of N-D tensors, reshapes them into 1-Dim flat tensor. """
    B = data.size(0)
    return data.view(B, -1)


def vectorize(data, dtype=torch.uint8):
    """ Vectorizes a tuple or int into a torch tensor. """
    # Don't need to vectorize an int, its broadcasted
    if isinstance(data, int):
        return data
    elif isinstance(data, tuple):
        # If the elements are the same, there is be no need for this, but nvm.
        return torch.tensor(data, dtype=dtype)
    else:
        raise ValueError(f"Invalid argument for vectorize. a must be either int or tuple of int.")


def assure_tensor(data, dtype):
    if not isinstance(data, torch.Tensor): data = torch.tensor(data, dtype=dtype)
    elif data.dtype != dtype: data = data.clone().detach().type(dtype)
    return data


def compute_output(dims, kernel_size, stride, padding=0):

    """ Computes the output shape given an input shape, kernel size, stride and padding. """

    dims = vectorize(dims)

    padding = vectorize(padding)
    kernel_size = vectorize(kernel_size)
    stride = vectorize(stride)

    # Vectorized computation
    dims = (dims - kernel_size + 2 * padding) / stride + 1

    return tuple(dims.numpy())


# TODO - Remove. Callbacks from now on
class TorchDataset(Dataset):

    def __init__(self, X, y, output_dtype, validation_split=0.0, X_val=None, y_val=None):

        super(TorchDataset, self).__init__()
        from sklearn.model_selection import train_test_split

        X, y = assure_tensor(X, torch.float32), assure_tensor(y, output_dtype)

        self.no_validation_data = True

        if validation_split > 0.0:
            if X_val is not None and y_val is not None:
                print("WARN: When passing validation_split argument dont pass X_val or y_val.", flush=True)
            X, X_val, y, y_val = train_test_split(X, y, test_size=validation_split)
            self.no_validation_data = False
        elif X_val is not None and y_val is not None:
            X_val, y_val = assure_tensor(X_val, torch.float32), assure_tensor(y_val, output_dtype)
            self.no_validation_data = False
        else:
            X_val, y_val = X, y

        self.X = X
        self.y = y

        self.X_val = X_val
        self.y_val = y_val

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
