from typing import Sequence, Optional

import numpy as np
from gym import Env
from gym.envs.registration import EnvSpec
from gym.spaces import Box, Discrete

import yaaf


class MarkovDecisionProcess(Env):

    def __init__(self, name: str,
                 states: Sequence[np.ndarray],
                 actions: Sequence[int],
                 transition_probabilities: np.ndarray,
                 rewards: np.ndarray,
                 discount_factor: float,
                 initial_state_distribution: np.ndarray,
                 state_meanings: Optional[Sequence[str]] = None,
                 action_meanings: Optional[Sequence[str]] = None,
                 terminal_states: Sequence[np.ndarray] = None):

        super(MarkovDecisionProcess, self).__init__()

        # MDP (S, A, P, R, gamma, miu)
        self._states = states
        self._num_states = len(states)
        self.state_meanings = state_meanings or ["<UNK>" for _ in range(self._num_states)]

        self._actions = actions
        self._num_actions = len(actions)
        self.action_meanings = action_meanings or ["<UNK>" for _ in range(self._num_actions)]

        self._P = transition_probabilities
        self._R = rewards
        self._discount_factor = discount_factor
        self._miu = initial_state_distribution

        # Metadata (OpenAI Gym)
        self.spec = EnvSpec(id=name)
        states_tensor = np.array(states).astype(np.float)
        self.observation_space = Box(
            low=states_tensor.min(),
            high=states_tensor.max(),
            shape=self._states[0].shape,
            dtype=states_tensor.dtype)
        self.action_space = Discrete(self._num_actions)
        self.reward_range = (rewards.min(), rewards.max())
        self.metadata = {}

        self._state = self.reset()
        self._terminal_states = terminal_states if terminal_states is not None else []

    # ########## #
    # OpenAI Gym #
    # ########## #

    def reset(self):
        x = np.random.choice(range(self.num_states), p=self._miu)
        initial_state = self._states[x]
        self._state = initial_state
        return initial_state

    def step(self, action):
        next_state = self.transition(self.state, action)
        reward = self.reward(self.state, action, next_state)
        is_terminal = self.is_terminal(next_state)
        self._state = next_state
        return next_state, reward, is_terminal, {}

    # ### #
    # MDP #
    # ### #

    def transition(self, state, action):
        x = self.state_index(state)
        y = np.random.choice(self.num_states, p=self.P[action, x])
        next_state = self.states[y]
        return next_state

    def reward(self, state, action, next_state):
        x = self.state_index(state) if not isinstance(state, int) else state
        y = self.state_index(next_state) if not isinstance(next_state, int) else next_state
        if self.R.shape == (self.num_states, self.num_actions): return self.R[x, action]
        elif self.R.shape == (self.num_states,): return self.R[y]
        elif self.R.shape == (self.num_states, self.num_actions, self.num_states): return self.R[x, action, y]
        else: raise ValueError("Invalid reward matrix R.")

    def is_terminal(self, state):
        return yaaf.ndarray_in_collection(state, self._terminal_states)

    @property
    def optimal_policy(self, method="policy iteration", **kwargs) -> np.ndarray:
        if not hasattr(self, "_pi"):
            greedy_q_value_tolerance = kwargs["greedy_q_value_tolerance"] if "greedy_q_value_tolerance" in kwargs else 10e-10
            if method == "policy iteration":
                self._pi = self.policy_iteration(greedy_q_value_tolerance)
            elif method == "value iteration":
                min_error = kwargs["min_error"] if "min_error" in kwargs else 10e-8
                V = self.value_iteration(min_error)
                Q = self.q_values(V)
                self._pi = self.extract_policy(Q, greedy_q_value_tolerance)
            else:
                raise ValueError(f"Invalid solution method for {self.spec.id} '{method}'")
        return self._pi

    @property
    def optimal_q_values(self) -> np.ndarray:
        if not hasattr(self, "_q_values"):
            V = self.optimal_values
            self._q_values = self.q_values(V)
        return self._q_values

    @property
    def optimal_values(self) -> np.ndarray:
        if not hasattr(self, "_values"):
            self._values = self.value_iteration()
        return self._values

    def value_iteration(self, min_error=1e-8):
        V = np.zeros(self.num_states)
        converged = False
        while not converged:
            Q = self.q_values(V)
            V_next = np.max(Q, axis=1)
            converged = np.linalg.norm(V - V_next) <= min_error
            V = V_next
        return V

    def policy_iteration(self, greedy_q_value_tolerance=1e-10):
        policy = np.ones((self.num_states, self.num_actions)) / self.num_actions
        converged = False
        while not converged:
            Q = self.policy_q_values(policy)
            next_policy = self.extract_policy(Q, greedy_q_value_tolerance)
            converged = (policy == next_policy).all()
            policy = next_policy
        return policy

    def policy_values(self, policy: np.ndarray):
        if policy.shape != (self.num_states, self.num_actions):
            raise ValueError(f"Invalid policy shape {policy.shape}. Policies for {self.spec.id} should have shape {(self.num_states, self.num_actions)}")
        R_pi = (policy * self.R).sum(axis=1)
        P_pi = np.zeros((self.num_states, self.num_states))
        for a in range(self.num_actions): P_pi += policy[:, a].reshape(-1, 1) * self.P[a]
        V_pi = np.linalg.inv(np.eye(self.num_states) - self.gamma * P_pi).dot(R_pi)
        return V_pi

    def q_values(self, values: np.ndarray):
        if values.shape != (self.num_states,):
            raise ValueError(f"Invalid values shape {values.shape}. Values for {self.spec.id} should have shape {(self.num_states,)}")
        values_as_column = values.reshape(-1, 1)
        Q = np.array([self.R[:, a].reshape(-1, 1) + self.gamma * self.P[a].dot(values_as_column) for a in range(self.num_actions)])[:, :, -1].T
        return Q

    def policy_q_values(self, policy: np.ndarray):
        if policy.shape != (self.num_states, self.num_actions):
            raise ValueError(f"Invalid policy shape {policy.shape}. Policies for {self.spec.id} should have shape {(self.num_states, self.num_actions)}")
        V_pi = self.policy_values(policy)
        Q_pi = self.q_values(V_pi)
        return Q_pi

    def extract_policy(self, q_values: np.ndarray, greedy_q_value_tolerance=10e-10) -> np.ndarray:
        if q_values.shape != (self.num_states, self.num_actions):
            raise ValueError(f"Invalid q-values shape {q_values.shape}. Q-Values for {self.spec.id} should have shape {(self.num_states, self.num_actions)}")
        Q_greedy = np.isclose(q_values, q_values.max(axis=1, keepdims=True), atol=greedy_q_value_tolerance, rtol=greedy_q_value_tolerance).astype(int)
        policy = Q_greedy / Q_greedy.sum(axis=1, keepdims=True)
        return policy

    # ########## #
    # Properties #
    # ########## #

    @property
    def state(self):
        return self._state

    @property
    def states(self):
        return self._states

    @property
    def num_states(self):
        return self._num_states

    @property
    def actions(self):
        return self._actions

    @property
    def num_actions(self):
        return self._num_actions

    @property
    def transition_probabilities(self):
        """Returns the Transition Probabilities P (array w/ shape X, X)"""
        return self._P

    @property
    def P(self):
        """Alias for self.transition_probabilities"""
        return self.transition_probabilities

    @property
    def rewards(self):
        """Returns the Rewards R (array w/ shape X, A)"""
        return self._R

    @property
    def R(self):
        """Alias for self.rewards"""
        return self.rewards

    @property
    def discount_factor(self):
        return self._discount_factor

    @property
    def gamma(self):
        """Alias for self.discount_factor"""
        return self._discount_factor

    @property
    def initial_state_distribution(self):
        return self._miu

    @property
    def miu(self):
        return self._miu

    @property
    def terminal_states(self):
        return self._terminal_states

    # ######### #
    # Auxiliary #
    # ######### #

    def state_index(self, state=None):
        """
            Returns the index of a given state in the state space.
            If the state is unspecified (None), returns the index of the current state st.
        """
        return self.state_index(self.state) if state is None else self.state_index_from(self.states, state)

    @staticmethod
    def state_index_from(states, state):
        """Returns the index of a state (array) in a list of states"""
        return yaaf.ndarray_index_from(states, state)

    # ########### #
    # Persistence #
    # ########### #

    @property
    def available_save_formats(self):
        return [
            "numpy",
            "numpy zip",
            "yaml"
        ]

    def save(self, directory, format="numpy"):
        if format == "numpy": self.save_numpy(directory)
        elif format == "numpy zip": self.save_numpy_zip(directory)
        elif format == "yaml": self.save_yaml(directory)
        else: raise ValueError(f"Invalid save format {format}. Please pick one of the following: {self.available_save_formats}")

    def save_numpy(self, directory):
        yaaf.mkdir(directory)
        np.save(f"{directory}/name", self.spec.id)
        np.save(f"{directory}/X", self.states)
        np.save(f"{directory}/A", self.actions)
        np.save(f"{directory}/P", self.transition_probabilities)
        np.save(f"{directory}/R", self.rewards)
        np.save(f"{directory}/gamma", self.gamma)
        np.save(f"{directory}/miu", self.initial_state_distribution)
        np.save(f"{directory}/X_meanings", self.state_meanings)
        np.save(f"{directory}/A_meanings", self.action_meanings)
        np.save(f"{directory}/X_terminal", self.terminal_states)

    def save_numpy_zip(self, directory):
        # TODO
        raise NotImplementedError()

    def save_yaml(self, directory):
        # TODO
        raise NotImplementedError()

    @staticmethod
    def load_numpy(directory):
        name = str(np.load(f"{directory}/name.npy"))
        states = np.load(f"{directory}/X.npy")
        actions = np.load(f"{directory}/A.npy")
        transition_probabilities = np.load(f"{directory}/P.npy")
        rewards = np.load(f"{directory}/R.npy")
        discount_factor = float(np.load(f"{directory}/gamma.npy"))
        initial_state_distribution = np.load(f"{directory}/miu.npy")
        state_meanings = tuple(np.load(f"{directory}/X_meanings.npy"))
        action_meanings = tuple(np.load(f"{directory}/A_meanings.npy"))
        terminal_states = np.load(f"{directory}/X_terminal.npy")
        mdp = MarkovDecisionProcess(
            name, states, actions, transition_probabilities, rewards, discount_factor, initial_state_distribution,
            state_meanings, action_meanings, terminal_states
        )
        return mdp

    @staticmethod
    def load_numpy_zip(directory):
        # TODO
        raise NotImplementedError()

    @staticmethod
    def load_yaml(directory):
        # TODO
        raise NotImplementedError()