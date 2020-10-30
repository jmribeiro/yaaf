import numpy as np
from yaaf import Timestep

from yaaf.agents import Agent
from yaaf.environments.markov import MarkovDecisionProcess


class GreedyMDPAgent(Agent):

    def __init__(self, mdp: MarkovDecisionProcess):
        self._mdp = mdp
        self._policy = mdp.optimal_policy
        super().__init__(f"MDP Agent for {mdp.spec.id}")

    def policy(self, state: np.ndarray):
        x = self._mdp.state_index(state)
        pi = self._policy[x]
        return pi

    def _reinforce(self, timestep: Timestep):
        pass