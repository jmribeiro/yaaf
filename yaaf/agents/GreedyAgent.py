from yaaf.agents import Agent
from yaaf.policies import greedy_policy


class GreedyAgent(Agent):

    """
    Greedy agent.
    Knows the mdp and follows its optimal policy.
    """

    def __init__(self, mdp):
        super().__init__(f"Greedy Agent for {mdp.spec.id}")
        self._q_star = mdp.q_values
        self._mdp = mdp

    def policy(self, observation):
        x = self._mdp.state_index(observation)
        q_values = self._q_star[x]
        return greedy_policy(q_values)

    def _reinforce(self, timestep):
        pass
