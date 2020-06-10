import numpy as np

from yaaf.evaluation import Metric


class AverageReturnMetric(Metric):

    def __init__(self, window=None):
        super(AverageReturnMetric, self).__init__("Average Return")
        self._rewards = []
        self._means = []
        self._stds = []
        self._window = window

    def reset(self):
        self._rewards = []
        self._means = []
        self._stds = []

    def __call__(self, timestep):
        reward = timestep.reward
        self._rewards.append(reward)
        mean, std = self._compute_stats()
        self._means.append(mean)
        self._stds.append(std)
        return self._means[-1]

    def _compute_stats(self):
        window = np.array(self._rewards)
        if self._window is not None:
            window = window[-self._window:]
        return window.mean(), window.std()

    def result(self):
        return np.array(self._means)



