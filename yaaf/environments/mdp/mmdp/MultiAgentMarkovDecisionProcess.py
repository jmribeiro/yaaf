from abc import ABC
from typing import Optional, Sequence

import numpy as np

from yaaf.environments.mdp import MarkovDecisionProcess as MDP
from yaaf.execution.Runner import Timestep


class MultiAgentMarkovDecisionProcess(MDP, ABC):

    def __init__(self, name: str, num_teammates: int,
                 state_space: Sequence[np.ndarray], disjoint_action_space: Sequence[int],
                 transition_probabilities: np.ndarray, rewards: np.ndarray,
                 discount_factor: float, initial_state_distribution: np.ndarray,
                 min_value_iteration_error: float = 10e-8,
                 action_meanings: Optional[Sequence[str]] = None):

        self._num_agents = num_teammates + 1
        self._num_disjoint_actions = len(disjoint_action_space)
        self._num_joint_actions = self._num_disjoint_actions ** self._num_agents

        joint_action_space = self._setup_joint_action_space(self._num_agents, disjoint_action_space)

        assert len(joint_action_space) == self._num_joint_actions

        super(MultiAgentMarkovDecisionProcess, self).__init__(name, state_space, joint_action_space,
                                                              transition_probabilities, rewards,
                                                              discount_factor, initial_state_distribution,
                                                              min_value_iteration_error, action_meanings)

        self._teammates = []

    def step(self, action):

        state = self.state
        teammates_actions = [teammate.action(state) for teammate in self._teammates]
        joint_actions = tuple([action] + teammates_actions)
        joint_action = self.action_space.index(joint_actions)
        next_state, reward, is_terminal, info = self.transition(state, joint_action)

        for i, action in enumerate(joint_actions):
            info[f"{self.spec.id} teammate {i} action"] = action

        timestep = Timestep(state, action, reward, next_state, is_terminal, info)
        [teammate.reinforcement(timestep) for teammate in self._teammates]

        return next_state, reward, is_terminal, {"joint actions": joint_actions}

    def optimal_disjoint_policy(self, agent_index):
        if not hasattr(self, "_optimal_disjoint_policy"):
            pi_star = self.policy
            self._optimal_disjoint_policy = np.zeros((self.num_states, self.num_disjoint_actions))
            for s in range(self.num_states):
                pi_s = pi_star[s]
                for a, action_probability in enumerate(pi_s):
                    optimal_joint_action = self.action_space[a]
                    optimal_action = optimal_joint_action[agent_index]
                    self._optimal_disjoint_policy[s, optimal_action] += action_probability
        return self._optimal_disjoint_policy

    @property
    def num_agents(self):
        return self._num_agents

    @property
    def num_teammates(self):
        return self._num_agents - 1

    @property
    def num_joint_actions(self):
        """ Alias for super().num_actions """
        return self.num_actions

    @property
    def joint_action_space(self):
        """ Alias for super().action_space """
        return self.action_space

    @property
    def num_disjoint_actions(self):
        return self._num_disjoint_actions

    def add_teammate(self, teammate):
        assert len(self._teammates) < self._num_agents - 1, "Maximum number of agents reached"
        self._teammates.append(teammate)

    @staticmethod
    def _setup_joint_action_space(num_agents, disjoint_action_space):

        joint_action_space = []

        for _ in range(num_agents):

            auxiliary = []

            if len(joint_action_space) == 0: # First action

                for a0 in range(len(disjoint_action_space)):
                    auxiliary.append([a0])

            else:

                for a in joint_action_space:

                    for a0 in range(len(disjoint_action_space)):
                        new_a = a + [a0]
                        auxiliary.append(tuple(new_a))

            joint_action_space = auxiliary

        return tuple(joint_action_space)
