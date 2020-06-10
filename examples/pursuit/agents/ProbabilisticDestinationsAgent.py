from examples.pursuit.agents import PursuitTeammate
from examples.pursuit.environment import PursuitState
from examples.pursuit.environment.utils import direction, move, agent_directions, manhattan_distance, softmax
import numpy as np

from yaaf.policies import deterministic_policy


class ProbabilisticDestinationsAgent(PursuitTeammate):

    def __init__(self, id, world_size):
        super().__init__("Probabilistic Destinations", id, world_size)

    def policy(self, observation):

        state = PursuitState.from_features(observation, self._world_size)

        directions = agent_directions()
        policy = np.zeros(4)
        w, h = state.world_size
        cell = state.agents_positions[self.id]
        prey = state.prey_positions[0]

        # don't go further than half the world
        maximum_lookout_distance = min(min(w, h) // 2, manhattan_distance(cell, prey, w, h))
        # if im next to the prey, move onto it
        if maximum_lookout_distance == 1:
            action = agent_directions().index(direction(cell, prey, w, h))
            return deterministic_policy(action, 4)

        distances = np.arange(1, maximum_lookout_distance)
        distance_probabilities = softmax(distances, -1)
        for i in range(len(distances)):
            dist = distances[i]
            # all destinations at distance = dist from the prey, which are unblocked and which action should the
            # agent take
            destinations, destination_directions = self.compute_destinations(dist, state)
            if len(destinations) == 0:
                continue

            # distances between each destination and me
            dist_to_me = np.array([manhattan_distance(cell, dest, w, h) for dest in destinations])

            dist_to_me_probs = softmax(dist_to_me, -1)
            for j, d in enumerate(destination_directions):
                policy[directions.index(d)] += dist_to_me_probs[j] * distance_probabilities[i]

        # if nothing available, move randomly to an unblocked cell
        if sum(policy) == 0:
            for d in directions:
                if move(cell, d, (w, h)) not in state.occupied_cells:
                    policy[directions.index(d)] = 1.0

        policy /= sum(policy)

        return policy

    def compute_destinations(self, distance, state):

        w, h = state.world_size
        px, py = state.prey_positions[0]
        my_pos = state.agents_positions[self.id]
        all_dests = []
        all_actions = []

        def destinations():
            # from top to right
            for i in range(distance):
                yield (px + i) % w, (py - distance + i) % h

            # from right to bottom
            for i in range(distance):
                yield (px + distance - i) % w, (py + i) % h

            # from bottom to left
            for i in range(distance):
                yield (px - i) % w, (py + distance - i) % h

            # from left to top
            for i in range(distance):
                yield (px - distance + i) % w, (py - i) % h

        for dest in destinations():
            action = direction(my_pos, dest, w, h)
            if dest not in state.occupied_cells and move(my_pos, action, (w, h)) not in state.occupied_cells:
                all_dests.append(dest)
                all_actions.append(action)

        return all_dests, all_actions
