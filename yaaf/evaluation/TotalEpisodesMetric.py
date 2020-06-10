from yaaf.evaluation import Metric


class TotalEpisodesMetric(Metric):

    def __init__(self):
        super().__init__("Total Episodes")
        self._episodes = 0

    def __call__(self, timestep):
        if timestep.is_terminal:
            self._episodes += 1

    def reset(self):
        self._episodes = 0

    def result(self):
        return self._episodes
