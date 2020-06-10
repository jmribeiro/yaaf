import numpy as np

from yaaf.evaluation import Metric


class PreyCapturesEveryTimestepIntervalMetric(Metric):

    def __init__(self, timestep_interval, verbose=False, log_interval=10):

        super(PreyCapturesEveryTimestepIntervalMetric, self).__init__(f"Prey Captures Every {timestep_interval} timesteps")

        self._timestep_interval = timestep_interval

        self._captures = []
        self._current_captures = 0
        self._timestep_counter = 0

        self._verbose = verbose
        self._log_interval = log_interval
        self._prev = 0

    def reset(self):
        self._captures = []
        self._current_captures = 0
        self._timestep_counter = 0
        self._prev = 0

    def __call__(self, timestep):

        self._timestep_counter += 1

        capture = timestep.is_terminal

        if capture:
            self._current_captures += 1

        if self._timestep_counter % self._timestep_interval == 0:
            self._captures.append(self._current_captures)
            self._current_captures = 0

        if self._verbose and self._timestep_counter % self._log_interval == 0:
            now = int(self._current_captures + np.array(self._captures).sum())
            print(f"{self._timestep_counter} timesteps. Prey captures: {now} (+{now-self._prev})", flush=True)
            self._prev = now

        return self._current_captures

    def result(self):
        return np.array(self._captures)
