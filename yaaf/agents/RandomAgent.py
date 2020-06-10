from yaaf.agents import Agent
from yaaf.policies import random_policy


class RandomAgent(Agent):

    def __init__(self, num_actions):
        super().__init__("Random Agent")
        self._num_actions = num_actions

    def policy(self, observation):
        return random_policy(self._num_actions)

    def _reinforce(self, timestep):
        pass
