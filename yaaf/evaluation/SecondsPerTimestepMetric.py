import time

import numpy as np

from yaaf.evaluation import Metric


class SecondsPerTimestepMetric(Metric):

    def __init__(self):
        super(SecondsPerTimestepMetric, self).__init__(f"Seconds Per Timestep")
        self._deltas = []
        self._last = None

    def reset(self):
        self._deltas = []

    def __call__(self, timestep):
        now = time.time()
        delta = now - self._last if self._last is not None else 0.0
        self._last = now
        self._deltas.append(delta)
        return delta

    def result(self):
        return np.array(self._deltas)
