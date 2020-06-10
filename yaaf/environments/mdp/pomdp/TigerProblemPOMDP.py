import random

from yaaf.environments.mdp.pomdp import PartiallyObservableMarkovDecisionProcess as POMDP
import numpy as np


class TigerProblemPOMDP(POMDP):

    def __init__(self, state_in_info=False):

        num_states = 2

        X = np.array([0, 1])
        A = np.array([0, 1, 2])
        Z = np.array([0, 1])

        P_Ol = P_Or = np.ones((num_states, num_states)) * 0.5
        P_L = np.eye(num_states)
        P = np.array([P_Ol, P_Or, P_L])

        O_Ol = O_Or = np.ones((num_states, num_states)) * 0.5
        O_L = np.array([[0.85, 0.15], [0.15, 0.85]])
        O = np.array([O_Ol, O_Or, O_L])

        R = np.array([
            [-1, 0, -0.1],
            [0, -1, -0.1]
        ])

        gamma = 0.95
        miu = np.ones(num_states) / num_states

        super(TigerProblemPOMDP, self).__init__("TigerProblemPOMDP-v0", X, A, Z, P, O, R, gamma, miu,
                                                action_meanings=["open the left door", "open the right door", "listen closely"],
                                                state_in_info=state_in_info)

        self._X_meanings = ["tiger behind the left door", "tiger behind the right door"]
        self._Z_meanings = ["hear a tiger on the left door", "hear a tiger on the right door"]
        self._last_action = None
        self._last_reward = None
        self._last_state = None

    def reset(self):
        self._state = random.choice(self.S)
        self._observation = random.choice(self.Z)
        self._last_action = None
        self._last_reward = None
        return self._observation

    def step(self, action):
        step = super().step(action)
        _, self._last_reward, _, _ = step
        self._last_action = action
        return step

    def render(self, mode="human"):

        if self._last_reward == 0.0:
            print(f"< There was no tiger behind the {self.action_meanings[self._last_action].split(' ')[2]} door! >")
            print(f"Congratulations, you escaped! [{self._last_reward}]")
        elif self._last_reward == -1.0:
            print(f"< There was a {self.state_meanings[self._state]} >")
            print(f"You died =( [{self._last_reward}]")
        else:
            print(f"< You seem to {self.observation_meanings[self._observation]} [{self._last_reward}]>")
            if self._last_action != 2 and self._last_action is not None:
                print(f"< There is a {self.state_meanings[self._state]} >")

    # ######### #
    # Auxiliary #
    # ######### #

    def is_terminal(self, state):
        return False

    def state_index(self, state=None):
        # Optimization - states in this POMDP match the indices, no need to search the state space
        return state

    # ########## #
    # Properties #
    # ########## #

    @property
    def state_meanings(self):
        return self._X_meanings

    @property
    def observation_meanings(self):
        return self._Z_meanings
