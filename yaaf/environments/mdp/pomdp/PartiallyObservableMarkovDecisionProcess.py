from abc import ABC
from typing import Sequence, Optional

import numpy as np
from yaaf.environments.mdp import MarkovDecisionProcess as MDP


class PartiallyObservableMarkovDecisionProcess(MDP, ABC):

    def __init__(self, name: str,
                 state_space: Sequence[np.ndarray],
                 action_space: Sequence[int],
                 observation_space: Sequence[np.ndarray],
                 transition_probabilities: np.ndarray,
                 observation_probabilities: np.ndarray,
                 rewards: np.ndarray,
                 discount_factor: float, initial_state_distribution: np.ndarray,
                 min_value_iteration_error: float = 10e-8,
                 action_meanings: Optional[Sequence[str]] = None,
                 state_in_info=False):

        self._observation_space = observation_space
        self._observation_probabilities = observation_probabilities

        self._observation = None

        self._state_in_info = state_in_info

        super(PartiallyObservableMarkovDecisionProcess, self).__init__(name,
                                                                       state_space, action_space,
                                                                       transition_probabilities, rewards,
                                                                       discount_factor, initial_state_distribution,
                                                                       min_value_iteration_error,
                                                                       action_meanings)

    def step(self, action):

        next_state, reward, is_terminal, info = super().step(action)

        if self._state_in_info:
            info["state"] = self._state
            info["next_state"] = next_state

        next_observation = self.observation(next_state, action)
        self._state = next_state
        self._observation = next_observation

        return next_observation, reward, is_terminal, info

    def observation(self, state, previous_action):
        observation_probabilities = self.O[previous_action, self.state_index(state)]
        observation = np.random.choice(self.Z, p=observation_probabilities)
        return observation

    # ########## #
    # Properties #
    # ########## #

    @property
    def Z(self):
        return self._observation_space

    @property
    def O(self):
        return self._observation_probabilities
