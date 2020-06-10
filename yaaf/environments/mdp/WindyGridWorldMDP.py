import numpy as np

from yaaf.environments.mdp import MarkovDecisionProcess as MDP


class WindyGridWorldMDP(MDP):

    def __init__(self, rows=7, columns=10,
                 start=(3, 0), goal=(3, 7),
                 wind=(0, 0, 0, 1, 1, 1, 2, 2, 1, 0),
                 discount_factor=0.99):

        assert len(wind) == columns, f"Wind must be tuple with length {columns}"

        X = tuple([np.array([x, y]) for x in range(rows) for y in range(columns)])
        A = tuple(range(4))

        self._wind = wind
        self._rows = rows
        self._columns = columns

        self._start_state = np.array(start)
        self._goal_state = np.array(goal)

        self.min_steps_to_solve = 15

        P = self.setup_transition_probabilities(X, A, wind)
        R = self.setup_rewards(X, A, self._goal_state)
        miu = np.zeros(len(X))
        miu[self.state_index_from(X, self._start_state)] = 1.0

        super(WindyGridWorldMDP, self).__init__("WindyGridWorldMDP-v0",
                                                X, A, P, R, discount_factor, miu,
                                                action_meanings=("Up", "Down", "Left", "Right"))

    def is_terminal(self, state):
        return np.array_equal(state, self._goal_state)

    # ######### #
    # Rendering #
    # ######### #

    def render(self, mode="human"):
        print(f"{self._draw_state()}")

    def _draw_state(self):

        state = ""

        columns = f" {' '.join([f' {i} ' for i in range(self._columns)])}\n"
        state += columns

        for row in range(self._rows):

            state += self._draw_row_border(self._columns)

            state += "|"
            for col in range(self._columns):
                cell = self._who_in(row, col)
                state += " " if col == 0 else "+ "
                state += f"{cell} "
            state += f"| {row}\n"

        state += self._draw_row_border(self._columns)
        state += f"-Boat Position: {self.state} ({self._wind[self.state[1]]}x wind)\n"

        return state

    def _who_in(self, row, column):

        boat_row, boat_column = self.state
        start_row, start_column = self._start_state
        goal_row, goal_column = self._goal_state
        wind = self._wind[column]

        if boat_row == row and boat_column == column:
            return 'B'
        elif row == start_row and column == start_column:
            return "S"
        elif row == goal_row and column == goal_column:
            return "G"
        elif wind != 0:
            return "^"
        else:
            return " "

    @staticmethod
    def _draw_row_border(columns):
        border = ""
        for y in range(columns):
            border += "+---"
        border += "+\n"
        return border

    # ############## #
    # MDP Structures #
    # ############## #

    @staticmethod
    def setup_transition_probabilities(state_space, action_space, wind):

        X = len(state_space)
        A = len(action_space)

        P = np.zeros((A, X, X))

        rows = state_space[-1][0] + 1
        columns = state_space[-1][1] + 1

        for s1 in range(X):

            state = state_space[s1]

            s1_transitions = dict()

            s1_transitions[0] = np.array([state[0] - wind[state[1]] - 1, state[1]])
            s1_transitions[1] = np.array([state[0] - wind[state[1]] + 1, state[1]])
            s1_transitions[2] = np.array([state[0] - wind[state[1]], state[1] - 1])
            s1_transitions[3] = np.array([state[0] - wind[state[1]], state[1] + 1])

            for action in range(A):
                s1_transitions[action][0] = max(min(s1_transitions[action][0], rows - 1), 0)
                s1_transitions[action][1] = max(min(s1_transitions[action][1], columns - 1), 0)
                next_state = s1_transitions[action]
                s2 = MDP.state_index_from(state_space, next_state)
                P[action][s1, s2] = 1.0

        return P

    @staticmethod
    def setup_rewards(state_space, action_space, goal_state):
        X = len(state_space)
        A = len(action_space)
        R = np.full((X, A), -1.0)
        R[MDP.state_index_from(state_space, goal_state), :] = 0.0
        return R
