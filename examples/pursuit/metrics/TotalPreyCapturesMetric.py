import numpy as np

from yaaf.evaluation import Metric


class TotalPreyCapturesMetric(Metric):

    def __init__(self):

        super(TotalPreyCapturesMetric, self).__init__(f"Total Prey Captures")

        self._captures = []
        self._current_captures = 0
        self._timestep_counter = 0

    def reset(self):
        self._captures = []
        self._current_captures = 0
        self._timestep_counter = 0

    def __call__(self, timestep):
        self._timestep_counter += 1
        capture = timestep.is_terminal
        if capture:
            self._current_captures += 1
        self._captures.append(self._current_captures)
        return self._current_captures

    def result(self):
        return np.array(self._captures)
