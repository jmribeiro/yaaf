from yaaf.evaluation import Metric
import numpy as np


class TimestepsPerEpisodeMetric(Metric):

    def __init__(self):
        super().__init__("Timesteps per Episode")
        self._timesteps = 0
        self._timesteps_per_episode = []

    def __call__(self, timestep):

        self._timesteps += 1

        if timestep.is_terminal:
            self._timesteps_per_episode.append(self._timesteps)
            self._timesteps = 0

        return self._timesteps

    def reset(self):
        self._timesteps = 0
        self._timesteps_per_episode.clear()

    def result(self):
        return np.array(self._timesteps_per_episode)



