import math

import numpy as np


class RunStats:

    def __init__(self, num_measures, confidence=0.99):
        self._num_measures = num_measures
        self._runs = []
        self._confidence = confidence
        self._means = np.zeros((num_measures,))
        self._std_devs = np.zeros((num_measures,))
        self._errors = np.zeros((num_measures,))
        self._N = 0

    def add(self, run):
        self._runs.append(run)
        self.recompute_errors(self._confidence)
        self._N = len(self._runs)

    def recompute_errors(self, confidence, max_n=math.inf):
        from yaaf.visualization import standard_error
        num_runs = len(self._runs)
        num_runs = min(max_n, num_runs)
        runs = np.array(self._runs[:num_runs])
        for measure in range(self._num_measures):
            column = runs[:, measure]
            self._means[measure] = column.mean()
            self._std_devs[measure] = column.std()
            self._errors[measure] = standard_error(self._std_devs[measure], num_runs, confidence)
            self._N = num_runs

    @property
    def confidence_level(self):
        return self._confidence

    @property
    def means(self):
        return self._means

    @property
    def errors(self):
        return self._errors

    @property
    def N(self):
        return self._N
