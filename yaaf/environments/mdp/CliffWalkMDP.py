import numpy as np

from yaaf.environments.mdp import MarkovDecisionProcess as MDP


class CliffWalkMDP(MDP):

    def __init__(self, rows=4, columns=12, discount_factor=1.0):

        X = tuple([np.array([x, y]) for x in range(columns) for y in range(rows)])
        A = tuple(range(4))

        num_states = len(X)
        num_actions = len(A)

        self._start_state = np.array((0, rows-1))
        self._goal_state = np.array((columns-1, rows-1))

        cliff_states = np.array([(column, rows-1) for column in range(1, columns-1)])
        start_state_index = MDP.state_index_from(X, self._start_state)
        goal_state_index = MDP.state_index_from(X, self._goal_state)
        cliff_state_indices = tuple([MDP.state_index_from(X, cliff) for cliff in cliff_states])

        miu = np.zeros(num_states)
        miu[start_state_index] = 1.0

        directions = np.array([(0, -1), (0, 1), (-1, 0), (1, 0)])

        P = self.setup_transition_probabilities(rows, columns, X, directions, start_state_index, goal_state_index, cliff_state_indices)
        R = self.setup_rewards(num_states, num_actions, cliff_state_indices, goal_state_index)

        self.min_steps_to_solve = 13

        super().__init__("CliffWalkMDP-v0",
                         X, A, P, R, discount_factor, miu,
                         action_meanings=("Up", "Down", "Left", "Right"))

    def is_terminal(self, state):
        return np.array_equal(state, self._goal_state)

    def render(self, mode="human"):
        pass

    # ############## #
    # MDP Structures #
    # ############## #

    @staticmethod
    def setup_transition_probabilities(rows, columns,
                                       state_space, directions,
                                       start_state_index, goal_state_index, cliff_state_indices):
        X = len(state_space)
        num_actions = len(directions)
        P = np.zeros((num_actions, X, X))
        for x in range(X):
            if x in cliff_state_indices:
                # Going to start
                cliff = x
                P[:, cliff, start_state_index] = 1.0
            elif x == goal_state_index:
                # Staying in same place
                P[:, x, x] = 1.0  # Not necessary due to restart
            else:
                column, row = state_space[x]
                for a in range(num_actions):
                    dx, dy = directions[a]
                    next_column = min(columns - 1, max(0, column + dx))
                    next_row = min(rows - 1, max(0, row + dy))
                    next_state = np.array([next_column, next_row])
                    y = MDP.state_index_from(state_space, next_state)
                    P[a, x, y] = 1.0
        return P

    @staticmethod
    def setup_rewards(num_states, num_actions, cliff_state_indices, goal_state_index):
        R = np.full((num_states, num_actions), fill_value=-1.0)
        R[cliff_state_indices, :] = -100
        R[goal_state_index, :] = 0.0  # Not necessary due to reset
        return R
