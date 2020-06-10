from yaaf.search import MCTSNode
import numpy as np


class PursuitMCTSNode(MCTSNode):

    def __init__(self, pursuit_state, transition_fn, teammates_fn):
        super().__init__(num_actions=4)
        self.pursuit_state = pursuit_state
        self._transition_fn = transition_fn
        self._teammates_fn = teammates_fn

    def __eq__(self, other):
        return isinstance(other, PursuitMCTSNode) and np.array_equal(self.pursuit_state.features(), other.pursuit_state.features())

    def simulate_action(self, action):
        joint_actions = self._joint_actions(action)
        next_state, reward = self._transition_fn(self.pursuit_state, joint_actions)
        reward = self._normalize_reward(reward)
        next_state_node = PursuitMCTSNode(next_state, self._transition_fn, self._teammates_fn)
        return next_state_node, reward, next_state.is_terminal

    def _joint_actions(self, action):
        teammates_actions = self._teammates_fn(self.pursuit_state)
        joint_actions = [action] + teammates_actions
        return joint_actions

    def _normalize_reward(self, reward):
        if reward == -1: reward = 0
        elif reward == 100: reward = 1
        else: raise ValueError(f"Invalid reward {reward}")
        return reward
