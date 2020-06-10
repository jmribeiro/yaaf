from yaaf.evaluation import Metric


class TotalTimestepsMetric(Metric):

    def __init__(self):
        super().__init__("Total Timesteps")
        self._timesteps = 0

    def __call__(self, timestep):
        self._timesteps += 1

    def reset(self):
        self._timesteps = 0

    def result(self):
        return self._timesteps
