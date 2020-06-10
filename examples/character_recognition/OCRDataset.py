from collections import defaultdict
from itertools import count
import torch
from torch.utils.data.dataset import Dataset

from yaaf.models.utils import unflatten


class OCRDataset(Dataset):

    def __init__(self, path="ocr_dataset/letter.data", val_fold=8, test_fold=9, image=False, binary=False):

        label_counter = count()
        labels = defaultdict(lambda: next(label_counter))
        X = []
        y = []
        fold = []
        with open(path) as f:
            for line in f:
                tokens = line.split()
                pixels = [int(t) for t in tokens[6:]]
                letter = labels[tokens[1]]
                fold.append(int(tokens[5]))
                X.append(pixels)
                if binary: letter = 0 if letter <= 13 else 1
                y.append(letter)

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        if image: X = unflatten(X, (1, 8, 16))

        fold = torch.tensor(fold, dtype=torch.long)

        train_idx = (fold != val_fold) & (fold != test_fold)
        dev_idx = fold == val_fold
        test_idx = fold == test_fold

        self.X = X[train_idx]
        self.y = y[train_idx]

        self.X_val = X[dev_idx]
        self.y_val = y[dev_idx]

        self.X_test = X[test_idx]
        self.y_test = y[test_idx]

        self.num_features = self.X.shape[1]
        self.num_classes = torch.unique(self.y).shape[0]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
