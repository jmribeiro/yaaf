from os.path import isdir
from unittest import TestCase

from examples.pursuit.agents import GreedyAgent, TeammateAwareAgent, ProbabilisticDestinationsAgent
from yaaf.agents import Agent
from examples.pursuit import Pursuit
from yaaf.execution import EpisodeRunner, TimestepRunner
from yaaf.models.feature_extraction import LinearSpec


class EnvironmentTests(TestCase):

    def _test_run(self, team, world_size):

        env = Pursuit(teammates=team, world_size=world_size)

        if team == "greedy":
            agent = GreedyAgent(0, env.world_size)
        elif team == "teammate aware":
            agent = GreedyAgent(0, env.world_size)
        elif team == "mixed":
            agent = ProbabilisticDestinationsAgent(0, env.world_size)
        else:
            raise ValueError(f"Invalid team {team}. Available teams: {env.available_teams()}.")

        EpisodeRunner(100, agent, env).run()

    def test_greedy_team_5_5(self):
        self._test_run("greedy", (5, 5))

    def test_teammate_aware_team_5_5(self):
        self._test_run("teammate aware", (5, 5))

    def test_mixed_team_5_5(self):
        self._test_run("mixed", (5, 5))


class PursuitAgentsTest(TestCase):

    def _test_agent_performance(self, agent: Agent,
                                teammates="teammate aware", features="default",
                                num_episodes=32, acceptable_avg_steps_til_capture=9.5):

        from yaaf.evaluation import TimestepsPerEpisodeMetric

        env = Pursuit(teammates=teammates, features=features)
        agent.eval()

        metric = TimestepsPerEpisodeMetric()
        EpisodeRunner(num_episodes, agent, env, observers=[metric]).run()

        average_steps_per_episode = metric.result().mean()
        assert average_steps_per_episode < acceptable_avg_steps_til_capture, \
            f"{agent.name} took, on average, {average_steps_per_episode} steps to capture the prey " \
            f"(required at most {acceptable_avg_steps_til_capture})"

        return average_steps_per_episode


class HandcodedAgentsTest(PursuitAgentsTest):

    def test_teammate_aware_agent(self):
        self._test_agent_performance(TeammateAwareAgent(id=0, world_size=(5, 5)), teammates="teammate aware")

    def test_greedy_agent(self):
        self._test_agent_performance(GreedyAgent(id=0, world_size=(5, 5)), teammates="greedy")

    def test_probabilistic_destinations_agent(self):
        self._test_agent_performance(ProbabilisticDestinationsAgent(id=0, world_size=(5, 5)),
                                     teammates="greedy", acceptable_avg_steps_til_capture=9.5)
        self._test_agent_performance(ProbabilisticDestinationsAgent(id=0, world_size=(5, 5)),
                                     teammates="teammate aware", acceptable_avg_steps_til_capture=13)


class DeepRLAgents(PursuitAgentsTest):

    def test_dqn_agent(self):

        """
        Tests if a trained DQN achieves <20 steps until capture
        and that an untrained DQN takes longer than that.
        """

        from yaaf.agents.dqn import MLPDQNAgent

        pretrained_dqn_saved = isdir(f"pretrained-agents/dqn_pursuit")

        env = Pursuit(teammates="teammate aware", features="relative agent")

        trained_dqn = MLPDQNAgent(env.num_features, env.num_actions)
        if pretrained_dqn_saved:
            trained_dqn.load(f"pretrained-agents/dqn_pursuit")
        else:
            TimestepRunner(5000, trained_dqn, env).run()
            trained_dqn.save(f"pretrained-agents/dqn_pursuit")

        untrained_dqn = MLPDQNAgent(env.num_features, env.num_actions)

        avg_steps_trained = self._test_agent_performance(
            agent=trained_dqn,
            teammates="teammate aware", features="relative agent",
            num_episodes=100,
            acceptable_avg_steps_til_capture=25
        )

        avg_steps_untrained = self._test_agent_performance(
            agent=untrained_dqn,
            teammates="teammate aware", features="relative agent",
            num_episodes=32,
            acceptable_avg_steps_til_capture=999
        )

        assert avg_steps_untrained > avg_steps_trained, \
            f"Untrained DQN achieving higher performance than trained DQN"

    def test_hga3c_agent(self):

        """
        Tests if a trained A3C (on GPU) achieves <20 steps until capture
        and that an untrained A3C (on GPU) takes longer than that.
        """

        from yaaf.execution import ParallelRunner
        from yaaf.agents.hga3c import HybridGA3CAgent

        PursuitA3C = lambda num_workers: HybridGA3CAgent(
            environment_names=["Pursuit" for _ in range(num_workers)],
            environment_actions=[4 for _ in range(num_workers)],
            observation_space=Pursuit(teammates="teammate aware", features="relative agent").observation_space.shape,
            conv2d_layers=[], mlp_layers=[LinearSpec(256, "relu"), LinearSpec(256, "relu")],
            learning_rate=0.001,
            discount_factor=0.95,
            start_threads=True
        )

        pretrained_hga3c_saved = isdir(f"pretrained-agents/hga3c_pursuit")
        num_workers = 16

        trained_hga3c = PursuitA3C(num_workers)
        if pretrained_hga3c_saved:
            trained_hga3c.load(f"pretrained-agents/hga3c_pursuit")
        else:
            envs = [
                Pursuit(teammates="teammate aware", features="relative agent")
                for _ in range(trained_hga3c.num_workers)
            ]
            ParallelRunner(trained_hga3c.workers, envs, max_timesteps=5000).run()
            trained_hga3c.save(f"pretrained-agents/hga3c_pursuit")

        untrained_hga3c = PursuitA3C(num_workers)

        avg_steps_trained = self._test_agent_performance(
            agent=trained_hga3c,
            teammates="teammate aware", features="relative agent",
            num_episodes=100,
            acceptable_avg_steps_til_capture=25
        )

        avg_steps_untrained = self._test_agent_performance(
            agent=untrained_hga3c,
            teammates="teammate aware", features="relative agent",
            num_episodes=32,
            acceptable_avg_steps_til_capture=999
        )

        assert avg_steps_untrained > avg_steps_trained, \
            f"Untrained HGA3C achieving higher performance than trained HGA3C"
