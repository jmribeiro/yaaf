from yaaf.policies import random_policy, policy_from_action
from examples.pursuit.environment import PursuitState
from examples.pursuit.agents import PursuitTeammate

from examples.pursuit.search import A_star_search
from examples.pursuit.environment.utils import distance, move, agent_directions, argmin, argmax, direction


class TeammateAwareAgent(PursuitTeammate):

    def __init__(self, id, world_size):
        super().__init__("Teammate Aware", id, world_size)
        self._last_prey_position = None
        self._prey_id = None
        self._last_target = None

    def policy(self, observation):

        state = PursuitState.from_features(observation, self._world_size)
        actions = agent_directions()
        num_actions = len(actions)
        position = state.agents_positions[self.id]
        w, h = state.world_size

        closest_prey, d, prey_id = None, None, 0

        for i, prey in enumerate(state.prey_positions):
            distance_to_prey = sum(distance(position, prey, w, h))
            # get the closest non cornered prey
            if d is None or (not state.cornered_position(state, prey, (w, h)) and distance_to_prey < d):
                closest_prey, d, prey_id = prey, distance_to_prey, i

        self._prey_id = prey_id
        self._last_prey_position = state.prey_positions[self._prey_id]

        agents = state.agents_positions

        # sort the agents by the worst shortest distance to the prey
        neighboring = [move(closest_prey, d, (w, h)) for d in actions]
        distances = [[sum(distance(a, p, w, h)) for p in neighboring] for a in agents]
        target_prey = 0
        for _ in range(len(agents)):
            min_distances = [min(d) for d in distances]
            min_indices = [argmin(d) for d in distances]
            selected_agent = argmax(min_distances)
            target_prey = min_indices[selected_agent]
            if selected_agent == self.id:
                break
            # remove the target from other agents
            for d in distances:
                d[target_prey] = 2 ** 31
            # remove the agent itself
            for i in range(len(distances[selected_agent])):
                distances[selected_agent][i] = -1

        self._last_target = neighboring[target_prey]

        target_prey = self._last_target
        # if already at destination, just follow the prey
        if position == target_prey:
            action = tuple(direction(position, self._last_prey_position, w, h))
            a = actions.index(action)
            return policy_from_action(a, num_actions)

        action, _ = A_star_search(position, state.occupied_cells - {target_prey}, target_prey, (w, h))

        if action is None:
            return random_policy(num_actions)
        else:
            a = actions.index(tuple(action))
            return policy_from_action(a, num_actions)