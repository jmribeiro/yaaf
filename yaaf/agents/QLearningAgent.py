from collections import defaultdict

import numpy as np

from yaaf.agents import Agent
from yaaf.policies import lazy_epsilon_greedy_policy


class QLearningAgent(Agent):

    def __init__(self, num_actions,
                 learning_rate=0.3, discount_factor=0.99, exploration_rate=0.15,
                 initial_q_values=0.0,
                 name="QLearning"):

        self._Q = defaultdict(lambda: np.ones(num_actions) * initial_q_values)
        self._learning_rate = learning_rate
        self._discount_factor = discount_factor
        self._training_exploration_rate = exploration_rate
        self._exploration_rate = exploration_rate
        self._num_actions = num_actions
        super(QLearningAgent, self).__init__(name)

    @property
    def Q(self):
        return self._Q

    def q_values(self, observation):
        x = tuple(observation) if len(observation.shape) > 0 else observation
        q_values = self._Q[x]
        return q_values

    def policy(self, observation):
        q_function = lambda: self.q_values(observation)
        return lazy_epsilon_greedy_policy(q_function, self._num_actions, self._exploration_rate)

    def _reinforce(self, timestep):

        s1 = tuple(timestep.observation) if len(timestep.observation.shape) > 0 else timestep.observation
        a = timestep.action
        r = timestep.reward
        s2 = tuple(timestep.next_observation) if len(timestep.next_observation.shape) > 0 else timestep.next_observation

        alpha = self._learning_rate
        gamma = self._discount_factor

        Q_s1_a = self._Q[s1][a]
        Q_s2_a = self._Q[s2]

        max_Q_s2_a = max(Q_s2_a)    # Off-Policy update (updates assuming greedy policy)

        self._Q[s1][a] = Q_s1_a + alpha * (r + gamma * max_Q_s2_a - Q_s1_a)

    def train(self):
        super().train()
        self._exploration_rate = self._training_exploration_rate

    def eval(self):
        super().eval()
        self._exploration_rate = 0.0

