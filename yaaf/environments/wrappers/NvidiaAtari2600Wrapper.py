from typing import Union

import gym
from gym import Wrapper, Env
from sklearn.externals._pilutil import imresize
import numpy as np
from queue import Queue


class NvidiaAtari2600Wrapper(Wrapper):

    """ A simpler version than all wrappers from open ai baselines """

    def __init__(self, env: Union[Env, Wrapper, str],
                 width: int = 84,
                 height: int = 84,
                 history_length: int = 4,
                 min_reward_clip: float = -1.0,
                 max_reward_clip: float = 1.0,
                 channels_first: bool = False):

        env = gym.make(env) if isinstance(env, str) else env

        super(NvidiaAtari2600Wrapper, self).__init__(env)

        self._name = self.unwrapped.spec.id
        self._num_actions = self.action_space.n
        self._frame_stack = Queue(maxsize=history_length)
        self._width, self._height = width, height
        self._channels_first = channels_first

        self.reward_range = (min_reward_clip, max_reward_clip)
        self.observation_space.shape = (width, height, history_length)

    @property
    def name(self):
        return self._name

    @property
    def num_actions(self):
        return self._num_actions

    def reset(self):
        self._frame_stack.queue.clear()
        stacked_frames = self._stack_frames(super().reset())
        while stacked_frames is None:
            stacked_frames, _, _, _ = self.step(0)
        return stacked_frames

    def step(self, action):
        observation, reward, done, info = super().step(action)
        next_observation = self._stack_frames(observation)
        info["Unclipped reward"] = reward
        clipped_reward = np.clip(reward, *self.reward_range)
        return next_observation, clipped_reward, done, info

    def _stack_frames(self, observation):

        # Already had full depth, remove the oldest
        if self._frame_stack.full(): self._frame_stack.get()

        # Add the new one
        self._frame_stack.put(self.preprocess(observation, self._width, self._height))

        # Game hasn't stacked enough frames yet
        if not self._frame_stack.full():
            return None
        else:
            # Stack state
            stacked_frames = np.array(self._frame_stack.queue)
            return stacked_frames if self._channels_first else np.transpose(stacked_frames, [1, 2, 0])

    @staticmethod
    def preprocess(observation, width, height):
        observation = np.dot(observation[..., :3], [0.299, 0.587, 0.114])
        observation = imresize(observation, [height, width], 'bilinear')
        observation = observation.astype(np.float32) / 128.0 - 1.0
        return observation

    def close(self):
        self._frame_stack.queue.clear()
        super().close()
