from unittest import TestCase

from yaaf.agents import GreedyAgent
from yaaf.environments.mdp import WindyGridWorldMDP
from yaaf.evaluation import TimestepsPerEpisodeMetric, AverageReturnMetric
from yaaf.evaluation import TotalTimestepsMetric
from yaaf.execution import EpisodeRunner


class MetricTests(TestCase):

    def test_timesteps_per_episode_metric(self):
        num_episodes = 1
        mdp = WindyGridWorldMDP()
        agent = GreedyAgent(mdp)
        metric = TimestepsPerEpisodeMetric()
        runner = EpisodeRunner(episodes=num_episodes, agent=agent, environment=mdp, observers=[metric])
        runner.run()
        result = metric.result()
        assert result.shape[0] == num_episodes
        assert result[0] == mdp.min_steps_to_solve

    def test_average_return_metric(self):
        num_episodes = 1
        mdp = WindyGridWorldMDP()
        agent = GreedyAgent(mdp)
        metric = AverageReturnMetric()
        runner = EpisodeRunner(episodes=num_episodes, agent=agent, environment=mdp, observers=[metric])
        runner.run()
        result = metric.result()
        assert result.shape[0] == mdp.min_steps_to_solve
        assert (result == -1).all()

    def test_total_timesteps_metric(self):
        num_episodes = 1
        mdp = WindyGridWorldMDP()
        agent = GreedyAgent(mdp)
        metric = TotalTimestepsMetric()
        runner = EpisodeRunner(episodes=num_episodes, agent=agent, environment=mdp, observers=[metric])
        runner.run()
        result = metric.result()
        assert result == mdp.min_steps_to_solve
