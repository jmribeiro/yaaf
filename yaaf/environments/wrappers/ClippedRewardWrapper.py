import numpy as np
from gym import Wrapper, Env


class ClippedRewardWrapper(Wrapper):

    def __init__(self, env: Env,
                 min_clip: float = -1.0,
                 max_clip: float = 1.0):
        super(ClippedRewardWrapper, self).__init__(env)
        self.reward_range = (min_clip, max_clip)

    def step(self, action):
        next_obs, reward, done, info = super().step(action)
        info["Unclipped reward"] = reward
        clipped_reward = np.clip(reward, *self.reward_range)
        return next_obs, clipped_reward, done, info
