from unittest import TestCase

from yaaf.agents import SARSAAgent, QLearningAgent
from yaaf.environments.markov import CliffWalkMDP
from yaaf.environments.markov.WindyGridWorldMDP import WindyGridWorldMDP
from yaaf.evaluation.TimestepsPerEpisodeMetric import TimestepsPerEpisodeMetric
from yaaf.execution import EpisodeRunner


class TraditionalReinforcementLearningTests(TestCase):

    @staticmethod
    def _test_agent_on_mdp(agent, mdp, minimum_steps_to_solve, acceptable_error=0):
        trainer = EpisodeRunner(500, agent, mdp)
        trainer.run()
        agent.eval()
        metric = TimestepsPerEpisodeMetric()
        evaluator = EpisodeRunner(100, agent, mdp, [metric])
        evaluator.run()
        mean = metric.result().mean()
        assert minimum_steps_to_solve - acceptable_error <= mean <= minimum_steps_to_solve + acceptable_error, \
            f"Expected between {minimum_steps_to_solve - acceptable_error} and {minimum_steps_to_solve + acceptable_error} but got {mean}"

    def test_q_learning_cliff_world(self):
        mdp = CliffWalkMDP()
        self._test_agent_on_mdp(QLearningAgent(mdp.num_actions), mdp, minimum_steps_to_solve=mdp.min_steps_to_solve)

    def test_q_learning_windy_grid_world(self):
        mdp = WindyGridWorldMDP()
        self._test_agent_on_mdp(QLearningAgent(mdp.num_actions), mdp, minimum_steps_to_solve=mdp.min_steps_to_solve)

    def test_sarsa_cliff_world(self):
        mdp = CliffWalkMDP()
        # Due to an on-policy update, when following an eps-greedy,
        # SARSA follows what Sutton calls a "safepath".
        # This means, to avoid random actions throwing it off the cliff, it goes as far away
        # as possible from the cliff, taking a total of 17 steps.
        self._test_agent_on_mdp(SARSAAgent(mdp.num_actions), mdp, minimum_steps_to_solve=17, acceptable_error=2)