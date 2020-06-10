from tqdm import tqdm


class ProgressLogger:

    def __init__(self, bottleneck, description="", count_episodes=False):
        self._reset = True
        self._bottleneck = bottleneck
        self._progress_bar = tqdm(range(bottleneck))
        self._progress_bar.set_description(description)
        self._count_episodes = count_episodes

    def __call__(self, timestep):
        if self._reset: self._reset = False
        if self._count_episodes and timestep.is_terminal:
            self._progress_bar.update(1)
        elif not self._count_episodes:
            self._progress_bar.update(1)

    def reset(self, description=""):
        if self._reset: return
        self._reset = True
        self._progress_bar = tqdm(range(self._bottleneck))
        self._progress_bar.set_description(description)


class TimestepsProgressLogger(ProgressLogger):

    def __init__(self, timesteps, description=""):
        super(TimestepsProgressLogger, self).__init__(timesteps, description)


class EpisodesProgressLogger(ProgressLogger):

    def __init__(self, episodes, description=""):
        super(EpisodesProgressLogger, self).__init__(episodes, description, count_episodes=True)
