import numpy as np

from yaaf.policies import random_policy
from examples.pursuit.environment import PursuitState
from examples.pursuit.agents import PursuitTeammate
from examples.pursuit.environment.utils import direction, direction_x, direction_y, distance, move


class GreedyAgent(PursuitTeammate):

    def __init__(self, id, world_size):
        super().__init__(f"Greedy {'Teammate' if id != 0 else 'Agent'}", id, world_size)

    def policy(self, observation):

        state = PursuitState.from_features(observation, self._world_size)
        policy = np.zeros((self._num_actions,))

        w, h = state.world_size
        my_pos = state.agents_positions[self.id]
        closest_prey, d = None, None
        for prey in state.prey_positions:

            distance_to_prey = sum(distance(my_pos, prey, w, h))

            # already neighboring some prey
            if distance_to_prey == 1:
                chosen_action = direction(my_pos, prey, w, h)
                action_index = self._action_space.index(tuple(chosen_action))
                policy[action_index] = 1.0
                return policy

            # get the closest non cornered prey
            if d is None or (not state.cornered_position(prey) and distance_to_prey < d):
                closest_prey, d = prey, distance_to_prey

        # unoccupied neighboring cells, sorted by proximity to agent
        targets = [move(closest_prey, dir, (w, h)) for dir in self._action_space]
        targets = list(filter(lambda x: x not in state.occupied_cells, targets))

        if len(targets) == 0:
            policy = random_policy(self._num_actions)
            return policy

        target = min(targets, key=lambda pos: sum(distance(my_pos, pos, w, h)))

        dx, dy = distance(my_pos, target, w, h)
        move_x = (direction_x(my_pos, target, w), 0)
        move_y = (0, direction_y(my_pos, target, h))
        pos_x = move(my_pos, move_x, (w, h))
        pos_y = move(my_pos, move_y, (w, h))

        # moving horizontally since there's a free cell
        if pos_x not in state.occupied_cells and (dx > dy or dx <= dy and pos_y in state.occupied_cells):
            action = move_x
            policy[self._action_space.index(action)] = 1.0
            return policy

        # moving vertically since there's a free cell
        elif pos_y not in state.occupied_cells and (dx <= dy or dx > dy and pos_x in state.occupied_cells):
            action = move_y
            policy[self._action_space.index(action)] = 1.0
            return policy

        # moving randomly since there are no free cells towards prey
        else:
            policy = random_policy(self._num_actions)
            return policy
