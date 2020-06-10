from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm


class MCTSNode(ABC):

    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.Q = np.zeros((self.num_actions,))
        self.N = np.ones((self.num_actions,))

    def uct_search(self, max_iterations, max_depth, exploration, discount_factor, verbose=False):
        """ Returns the best action using Monte Carlo Tree Search. """
        visited_nodes = []
        iterator = tqdm(range(max_iterations)) if verbose else range(max_iterations)
        for _ in iterator:
            self.simulate(max_depth, exploration, discount_factor, visited_nodes)
        return self.Q.argmax()

    def simulate(self, depth, exploration, discount_factor, visited_nodes):

        if depth == 0:
            return 0.0

        if self not in visited_nodes:
            visited_nodes.append(self)
            for action in range(self.num_actions):
                self.expand(action, depth, discount_factor)
            node = self
        else:
            node = visited_nodes[visited_nodes.index(self)]

        action = node.upper_confidence_bound(exploration).argmax()
        q = self.expand(action, depth, discount_factor)

        return q

    def expand(self, action, depth, discount_factor):

        next_node, reward, is_terminal = self.simulate_action(action)

        if is_terminal:
            q = reward
        else:
            q = reward + discount_factor * next_node.rollout(depth - 1, discount_factor)

        self.N[action] += 1
        self.Q[action] += (q - self.Q[action]) / self.N[action]

        return q

    def rollout(self, depth, discount_factor):
        """Monte Carlo Rollout using a Random Policy"""
        if depth == 0:
            return 0.0
        action = np.random.choice(range(self.num_actions))
        next_state_node, reward, is_terminal = self.simulate_action(action)
        if is_terminal:
            return reward
        else:
            return reward + discount_factor * next_state_node.rollout(depth - 1, discount_factor)

    def upper_confidence_bound(self, Cp):
        """Vectorized Implementation of the UCB Formula! (NumPy)"""
        return self.Q + Cp * np.sqrt(2 * np.log(self.N.sum()) / self.N)

    @abstractmethod
    def __eq__(self, other):
        """Returns a unique identifier for an MCTS node."""
        raise NotImplementedError()

    def simulate_action(self, action):
        raise NotImplementedError()
