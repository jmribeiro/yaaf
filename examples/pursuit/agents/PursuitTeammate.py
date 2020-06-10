from abc import ABC

from yaaf.agents import Agent
from examples.pursuit.environment.utils import agent_directions


class PursuitTeammate(Agent, ABC):

    def __init__(self, name, id, world_size):
        super().__init__(name)
        self._id = id
        self._action_space = agent_directions()
        self._num_actions = len(self._action_space)
        self._world_size = world_size

    @property
    def id(self):
        return self._id

    def _reinforce(self, timestep):
        pass
