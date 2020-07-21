from yaaf.search import MCTSNode


class MDPMCTSNode(MCTSNode):

    """
        MCTS Node for an MDP.
    """

    def __init__(self, state, mdp, is_terminal=False):
        super().__init__(num_actions=mdp.num_actions)
        self.state = state
        self.markov = mdp
        self._is_terminal = is_terminal

    def __eq__(self, other):
        return self.markov.state_index(self.state)

    def simulate_action(self, action):
        next_state, reward, is_terminal, info = self.markov.transition(self.state, action)
        return MDPMCTSNode(next_state, self.markov, is_terminal), reward, is_terminal

    @property
    def is_terminal(self):
        return self._is_terminal
