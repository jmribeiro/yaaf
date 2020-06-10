from yaaf.evaluation import AverageReturnMetric


class AverageEpisodeReturnMetric(AverageReturnMetric):

    def __init__(self, window=None):
        super(AverageEpisodeReturnMetric, self).__init__(window)
        self._episode_reward = 0
        self._name = "Average Episode Return"

    def __call__(self, timestep):

        terminal = timestep.is_terminal
        reward = timestep.reward
        self._episode_reward += reward

        if terminal:
            self._rewards.append(self._episode_reward)
            self._episode_reward = 0
            mean, std = self._compute_stats()
            self._means.append(mean)
            self._stds.append(std)

        return self._means[-1] if len(self._means) > 0 else 0.0
