from yaaf.search import MCTSNode


class MDPMCTSNode(MCTSNode):

    """
        MCTS Node for an MDP.
    """

    def __init__(self, state, mdp, is_terminal=False):
        super().__init__(num_actions=mdp.A)
        self.state = state
        self.mdp = mdp
        self._is_terminal = is_terminal

    def __eq__(self, other):
        return self.mdp.state_index(self.state)

    def simulate_action(self, action):
        next_state, reward, is_terminal, info = self.mdp.transition(self.state, action)
        return MDPMCTSNode(next_state, self.mdp, is_terminal), reward, is_terminal

    @property
    def is_terminal(self):
        return self._is_terminal
