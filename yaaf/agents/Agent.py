from abc import ABC, abstractmethod
from multiprocessing import Value

from numpy.core.multiarray import ndarray

from yaaf import Timestep
from yaaf import mkdir, isdir
from yaaf.policies import action_from_policy


class Agent(ABC):

    """
    Base agent class.
    Represents the concept of an autonomous agent.
    """

    def __init__(self, name: str):
        """
        Constructor.
        :parameter name (str) - The name of the agent.
        """
        self._name = name
        self._trainable = Value("b", True)
        self._total_training_timesteps = Value("i", 0)

    # ######### #
    # Interface #
    # ######### #

    def action(self, observation: ndarray):
        """
        Returns the agent's action for a given observation of the environment.
        Default implementation for choosing an action (override if necessary).
        Calls the agent's policy method, obtaining the policy (ndarray, prob. distribution)
        and randomly samples an action taking the probabilities into account.
        :parameter observation (ndarray) - Observation of the environment to act upon.
        :returns action (int/float) - Action to execute upon the environment.
        """
        policy = self.policy(observation)
        action = action_from_policy(policy, not self.trainable)
        return action

    @abstractmethod
    def policy(self, observation: ndarray):
        """
        Returns the agent's policy for a given observation of the environment.
        :parameter observation (ndarray) - Observation of the environment used to select an action.
        :returns policy (ndarray) - A distribution over possible actions.
        """
        raise NotImplementedError()

    def reinforcement(self, timestep: Timestep):

        """
        Provides reinforcement for the agent.
        :parameter timestep (Timestep) - A named tuple containing:
            observation (ndarray) - Observation of the environment before transitioning.
            action (int) - Action executed upon the environment.
            reward (float) - Reward obtained by transitioning.
            next_observation (ndarray) - Observation of the environment after transitioning.
            is_terminal (bool) - True if episode ended after transitioning.
            info (dict) - Additional information from the environment.
        """

        if self.trainable:
            self._total_training_timesteps.value += 1
            agent_info = self._reinforce(timestep) or {}
            timestep.info[self.name] = agent_info
            return agent_info

    @abstractmethod
    def _reinforce(self, timestep: Timestep):
        """
        Template Method for self.reinforce
        :returns info (Optional[None, dict]) - Any relevant info regarding updates (such as model losses, etc...)
        """
        raise NotImplementedError()

    def train(self):
        """
        Enables the agent's training mode.
        When in training mode, the agent can learn from new timesteps (given through the reinforcement method).
        """
        self._trainable.value = True

    def eval(self):
        """
        Enables the agent's evaluation mode.
        When in evaluation mode, the agent doesn't learn from new timesteps (even if given through the reinforcement method).
        """
        self._trainable.value = False

    # ########### #
    # Persistence #
    # ########### #

    def save(self, directory: str):
        """
        Saves the agent's state into to a directory.
        :parameter directory (str) - Path of the directory to save the agent's state into.
        """
        mkdir(directory)

    def load(self, directory: str):
        """
        Loads an agent's state from a given directory.
        :parameter directory (str) - Path of the directory to save the agent's state into.
        """
        if not isdir(directory):
            raise ValueError(f"Agent save directory {directory} does not exist.")

    # ########## #
    # Properties #
    # ########## #

    @property
    def name(self):
        """
        Property (str). The agent's name.
        """
        return self._name

    @property
    def trainable(self):
        """
        Property (bool). Flag indicating if the agent is in training mode.
        """
        return self._trainable.value

    @property
    def total_training_timesteps(self):
        """
        Property (int). The total number of timesteps (or "frames") given to the agent through the reinforcement method.
        """
        return self._total_training_timesteps.value

    @property
    def params(self):
        """
        Property (dict). Any relevant information regarding the agent. Override if necessary
        """
        return dict(name=self.name)
