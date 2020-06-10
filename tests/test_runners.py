import time
from unittest import TestCase, main

from yaaf.agents import RandomAgent
from yaaf.environments.wrappers import NvidiaAtari2600Wrapper
from yaaf.evaluation import TotalEpisodesMetric
from yaaf.execution import EpisodeRunner, TimeLimitRunner, TimestepRunner


class RunnerTests(TestCase):

    def test_episode_runner(self):
        env = NvidiaAtari2600Wrapper("Breakout-v0")
        agent = RandomAgent(env.num_actions)
        counter = TotalEpisodesMetric()
        runner = EpisodeRunner(1, agent, env, observers=[counter])
        runner.run()
        assert counter.result() == 1

    def test_timestep_runner(self):
        import gym
        env = gym.make("Breakout-v0")
        agent = RandomAgent(env.action_space.n)
        runner = TimestepRunner(1000, agent, env)
        runner.run()
        assert agent.total_training_timesteps == 1000

    def test_time_limit_runner(self):
        import gym
        seconds = 3
        acceptable_margin = 5e-2
        env = gym.make("Breakout-v0")
        agent = RandomAgent(env.action_space.n)
        runner = TimeLimitRunner(seconds, agent, env)
        start = time.time()
        runner.run()
        end = time.time()
        total = end - start
        assert total <= seconds + acceptable_margin


if __name__ == '__main__':
    main()
