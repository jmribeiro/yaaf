import random

import numpy as np

from examples.pursuit.environment.utils import neighbors, prey_directions, direction_x, distance, direction_y


class PursuitState:

    def __init__(self, agent_positions, prey_positions, world_size):

        assert (isinstance(agent_positions, tuple))
        assert (isinstance(prey_positions, tuple))
        assert (isinstance(world_size, tuple))
        assert (len(world_size) == 2)

        self._agent_positions = agent_positions
        self.num_agents = len(agent_positions)
        self._prey_positions = prey_positions
        self._terminal = True
        self._world_size = world_size
        self._occupied = None

        for prey in prey_positions:
            if not self.cornered_position(prey):
                self._terminal = False
                break

    # ########## #
    # Properties #
    # ########## #

    @property
    def agents_positions(self):
        return self._agent_positions

    @property
    def prey_positions(self):
        return self._prey_positions

    @property
    def is_terminal(self):
        return self._terminal

    @property
    def world_size(self):
        return self._world_size

    # ######## #
    # Features #
    # ######## #

    def features(self):
        return np.concatenate((np.array(self._agent_positions), np.array(self._prey_positions))).reshape(-1)

    def features_relative_prey(self):
        prey = self._prey_positions[0]
        relative_pos = []
        w, h = self._world_size
        for pos in self._agent_positions:
            relative_pos.append(self.distance_to(prey, pos, w, h))
        features = np.concatenate(relative_pos).reshape(-1)
        return features

    def features_relative_agent(self, agent_id):

        feature_array = []
        agent = self._agent_positions[agent_id]
        w, h = self._world_size

        # Teammates
        for i, teammate in enumerate(self._agent_positions):
            if i == agent_id: continue
            d = self.distance_to(agent, teammate, w, h)
            feature_array.append(d)

        # Sort (from closest to furthest)
        feature_array.sort(key=lambda dists: sum(abs(dists)))

        # Prey
        prey = self._prey_positions[0]
        d = self.distance_to(agent, prey, w, h)
        feature_array.append(d)

        return np.concatenate(feature_array).reshape(-1)

    # ######### #
    # Auxiliary #
    # ######### #

    @staticmethod
    def distance_to(pivot, other, w, h):
        dx, dy = distance(pivot, other, w, h)
        dx = dx * direction_x(pivot, other, w)
        dy = dy * direction_y(pivot, other, h)
        return np.array([dx, dy])

    @staticmethod
    def move_prey_randomly():
        possible_prey_directions = prey_directions()
        return random.choice(possible_prey_directions)

    def cornered_position(self, position):
        for neighbor in neighbors(position, self._world_size):
            if neighbor not in self.occupied_cells:
                return False
        return True

    @property
    def occupied_cells(self):
        if not self._occupied: self._occupied = set(self._agent_positions) | set(self._prey_positions)
        return self._occupied

    # ######### #
    # Operators #
    # ######### #

    def __add__(self, other):
        assert (isinstance(other, np.ndarray))
        features = self.features()
        new_features = (features + other) % ([self._world_size[0], self._world_size[1]] * (len(features) // 2))
        return PursuitState.from_features(new_features, self._world_size)

    def __sub__(self, other):
        assert (isinstance(other, PursuitState))
        features = self.features()
        max_list = [self._world_size[0], self._world_size[1]] * (len(features) // 2)
        half_list = [value // 2 for value in max_list]
        return ((features - other.features()) + half_list) % max_list - half_list

    def __radd__(self, other):
        return self.__add__(other)

    def __hash__(self):
        return hash(self._agent_positions)

    def __eq__(self, other):
        return self.agents_positions == other.agents_positions and \
               self.prey_positions == other.prey_positions and \
               self.world_size == other.world_size

    def __repr__(self):

        state = ""

        Y = self.world_size[0]
        X = self.world_size[1]

        # Draw Row Border

        for y in range(Y):

            state += self._draw_row_border(X)

            y_0 = self._who_in(0, y)

            state += "|"
            state += f" {y_0} "
            for x in range(X - 1):
                state += f"+ {self._who_in(x + 1, y)} "
            state += "|\n"

        state += self._draw_row_border(X)
        return state

    def _who_in(self, x, y):

        agents = self.agents_positions
        prey = self.prey_positions[0]

        if prey[0] == x and prey[1] == y:
            return "p"

        for i, (xa, ya) in enumerate(agents):
            if x == xa and ya == y:
                return i

        return " "

    @staticmethod
    def _draw_row_border(columns):
        border = ""
        for y in range(columns):
            border += "+---"
        border += "+\n"
        return border

    @staticmethod
    def random_state(num_agents, world_size):

        assert (num_agents >= 4)
        world_size = tuple(world_size)
        num_preys = num_agents // 4

        assert (world_size[0] * world_size[1] > num_agents + num_preys)
        filled_positions = set()

        prey_positions = [(0, 0)] * num_preys
        agents_positions = [(0, 0)] * num_agents

        prey_positions, filled_positions = PursuitState._process(num_preys, world_size, prey_positions, filled_positions)
        agents_positions, filled_positions = PursuitState._process(num_agents, world_size, agents_positions, filled_positions)

        return PursuitState(prey_positions=tuple(prey_positions), agent_positions=tuple(agents_positions), world_size=world_size)

    @staticmethod
    def _process(num_entities, world_size, positions, filled_positions):
        for e in range(num_entities):
            while True:
                position = (np.random.randint(0, world_size[0] - 1), np.random.randint(0, world_size[1] - 1))
                if position not in filled_positions:
                    break
            positions[e] = position
            filled_positions.add(position)
        return positions, filled_positions

    @staticmethod
    def from_features(features, world_size):
        features = features.reshape(-1, 2)
        agent_positions = tuple(tuple(pos) for pos in features[:4])
        prey_position = (tuple(features[4]),)
        return PursuitState(agent_positions=agent_positions, prey_positions=prey_position, world_size=world_size)

    @staticmethod
    def from_features_relative_prey(features, world_size):
        agent_positions = features.reshape(-1, 2)
        mid = (world_size[0] // 2, world_size[1] // 2)
        for i, (x, y) in enumerate(agent_positions):
            agent_positions[i] = (mid[0] + x, mid[1] + y)
        agent_positions = tuple(tuple(pos) for pos in agent_positions)
        return PursuitState(agent_positions=agent_positions, prey_positions=(mid,), world_size=world_size)