from unittest import TestCase, main

import numpy as np

from yaaf.agents import SARSAAgent, QLearningAgent, GreedyAgent
from yaaf.environments.mdp import CliffWalkMDP
from yaaf.environments.mdp.WindyGridWorldMDP import WindyGridWorldMDP
from yaaf.evaluation.TimestepsPerEpisodeMetric import TimestepsPerEpisodeMetric
from yaaf.execution import EpisodeRunner


class MDPSolverTests(TestCase):

    def _test_mdp_greedy_policy(self, mdp, episodes=100):
        agent = GreedyAgent(mdp)
        metric = TimestepsPerEpisodeMetric()
        runner = EpisodeRunner(episodes, agent, mdp, [metric])
        runner.run()
        assert metric.result().mean() == mdp.min_steps_to_solve

    def test_windy_grid_world_optimal_policy(self):
        self._test_mdp_greedy_policy(WindyGridWorldMDP(), episodes=1)

    def test_cliff_walk_optimal_policy(self):
        self._test_mdp_greedy_policy(CliffWalkMDP(), episodes=1)

    def test_windy_grid_world_value_iteration(self):

        """This test compares the solution from the MDP interface with the WindyGridWorld's solution."""

        mdp = WindyGridWorldMDP()
        q_values = mdp.q_values  # keep arguments due to expected q_star
        expected_q_values = np.array([
            [-88.97864878, -88.97864878, -88.97864878, -88.86732201],
            [-88.86732201, -88.86732201, -88.97864878, -88.75487073],
            [-88.75487073, -88.75487073, -88.86732201, -88.64128358],
            [-88.64128358, -88.64128358, -88.75487073, -88.52654908],
            [-88.52654908, -88.52654908, -88.64128358, -88.41065565],
            [-88.41065565, -88.41065565, -88.52654908, -88.29359158],
            [-88.29359158, -88.29359158, -88.41065565, -88.17534503],
            [-88.17534503, -88.17534503, -88.29359158, -88.05590408],
            [-88.05590408, -88.05590408, -88.17534503, -87.93525666],
            [-87.93525666, -87.81339058, -88.05590408, -87.93525666],
            [-88.97864878, -88.97864878, -88.97864878, -88.86732201],
            [-88.86732201, -88.86732201, -88.97864878, -88.75487073],
            [-88.75487073, -88.75487073, -88.86732201, -88.64128358],
            [-88.64128358, -88.64128358, -88.75487073, -88.52654908],
            [-88.52654908, -88.52654908, -88.64128358, -88.41065565],
            [-88.41065565, -88.41065565, -88.52654908, -88.29359158],
            [-88.29359158, -88.29359158, -88.41065565, -88.17534503],
            [-88.17534503, -88.17534503, -88.29359158, -88.05590408],
            [-88.05590408, -88.05590408, -88.17534503, -87.93525666],
            [-87.93525666, -87.69029353, -88.05590408, -87.81339058],
            [-88.97864878, -88.97864878, -88.97864878, -88.86732201],
            [-88.86732201, -88.86732201, -88.97864878, -88.75487073],
            [-88.75487073, -88.75487073, -88.86732201, -88.64128358],
            [-88.64128358, -88.64128358, -88.75487073, -88.52654908],
            [-88.52654908, -88.52654908, -88.64128358, -88.41065565],
            [-88.41065565, -88.41065565, -88.52654908, -88.29359158],
            [-88.29359158, -88.29359158, -88.41065565, -88.17534503],
            [-88.17534503, -88.17534503, -88.29359158, -88.05590408],
            [-88.05590408, -87.93525666, -88.17534503, -87.81339058],
            [-87.81339058, -87.56595307, -87.93525666, -87.69029353],
            [-88.97864878, -88.97864878, -88.97864878, -88.86732201],
            [-88.86732201, -88.86732201, -88.97864878, -88.75487073],
            [-88.75487073, -88.75487073, -88.86732201, -88.64128358],
            [-88.64128358, -88.64128358, -88.75487073, -88.52654908],
            [-88.52654908, -88.52654908, -88.64128358, -88.41065565],
            [-88.41065565, -88.41065565, -88.52654908, -88.29359158],
            [-88.29359158, -88.29359158, -88.41065565, -88.17534503],
            [-87.17534503, -87.17534503, -87.29359158, -87.05590408],
            [-88.05590408, -87.81339058, -88.17534503, -87.69029353],
            [-87.69029353, -87.44035665, -87.81339058, -87.56595307],
            [-88.97864878, -88.97864878, -88.97864878, -88.86732201],
            [-88.86732201, -88.86732201, -88.97864878, -88.75487073],
            [-88.75487073, -88.75487073, -88.86732201, -88.64128358],
            [-88.64128358, -88.64128358, -88.75487073, -88.52654908],
            [-88.52654908, -88.52654908, -88.64128358, -88.41065565],
            [-88.41065565, -88.41065565, -88.52654908, -88.29359158],
            [-88.29359158, -88.29359158, -88.41065565, -88.17534503],
            [-88.17534503, -87.18534503, -88.29359158, -87.93525666],
            [-87.93525666, -87.31349158, -87.18534503, -87.56595307],
            [-87.56595307, -87.56595307, -87.31349158, -87.44035665],
            [-88.97864878, -88.97864878, -88.97864878, -88.86732201],
            [-88.86732201, -88.86732201, -88.97864878, -88.75487073],
            [-88.75487073, -88.75487073, -88.86732201, -88.64128358],
            [-88.64128358, -88.64128358, -88.75487073, -88.52654908],
            [-88.52654908, -88.52654908, -88.64128358, -88.41065565],
            [-88.41065565, -88.41065565, -88.52654908, -88.29359158],
            [-88.29359158, -88.29359158, -88.41065565, -87.18534503],
            [-88.17534503, -87.31349158, -88.29359158, -87.81339058],
            [-87.81339058, -87.44035665, -87.31349158, -87.44035665],
            [-87.44035665, -87.56595307, -87.44035665, -87.56595307],
            [-88.97864878, -88.97864878, -88.97864878, -88.86732201],
            [-88.86732201, -88.86732201, -88.97864878, -88.75487073],
            [-88.75487073, -88.75487073, -88.86732201, -88.64128358],
            [-88.64128358, -88.64128358, -88.75487073, -88.52654908],
            [-88.52654908, -88.52654908, -88.64128358, -88.41065565],
            [-88.41065565, -87.44035665, -88.52654908, -87.31349158],
            [-88.29359158, -87.31349158, -88.41065565, -87.31349158],
            [-87.18534503, -87.44035665, -88.29359158, -87.31349158],
            [-87.31349158, -87.44035665, -87.44035665, -87.56595307],
            [-87.56595307, -87.56595307, -87.44035665, -87.56595307]
        ])
        assert np.array_equal(q_values.round(4), expected_q_values.round(4))


class ReinforcementLearningTests(TestCase):

    def _test_agent_on_mdp(self, agent, mdp, minimum_steps_to_solve, acceptable_error=0):
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


if __name__ == '__main__':
    main()
