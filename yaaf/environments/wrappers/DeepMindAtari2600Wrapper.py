from typing import Optional, Union

import gym
from gym import Wrapper, Env
from gym.wrappers import AtariPreprocessing, TimeLimit, FrameStack
import numpy as np


class DeepMindAtari2600Wrapper(Wrapper):

    def __init__(self, env: Union[Env, Wrapper, str],
                 noop_max: int = 30,
                 frameskip: int = 4,
                 stacked_frames: int = 4,
                 screen_size: int = 84,
                 grayscale_obs: bool = True,
                 scale_obs: bool = False,
                 clip_reward: bool = True,
                 min_reward_clip: float = -1.0,
                 max_reward_clip: float = 1.0,
                 terminal_on_life_loss: bool = True,
                 max_steps_per_episode: Optional[int] = None,
                 compress_frames=True):

        if isinstance(env, str):
            if "NoFrameskip" in env:
                env = gym.make(env)
            else:
                env = gym.make(env, frameskip=1)
                env.spec.id = f"{env.spec.id}(NoFrameskip)"

        # Main Wrapping (OpenAI)
        # > No Ops
        # > Explicit frameskipping
        # > Resized
        # > Terminal on ale lives -= 1 or == 0
        # > Grayscaled
        # > Scaled
        env = AtariPreprocessing(env, noop_max, frameskip, screen_size, terminal_on_life_loss, grayscale_obs, scale_obs)

        # Time (Step) Limit Wrapper
        if max_steps_per_episode is not None:
            env = TimeLimit(env, max_episode_steps=max_steps_per_episode)

        # Fire to reset
        if 'FIRE' in env.unwrapped.get_action_meanings():
            from yaaf.environments.wrappers import FireToResetWrapper
            env = FireToResetWrapper(env)

        # Clipped reward -1, 0, +1
        if clip_reward:
            from yaaf.environments.wrappers import ClippedRewardWrapper
            env = ClippedRewardWrapper(env, min_reward_clip, max_reward_clip)

        # Stacked frames for history length
        if stacked_frames is not None:
            env = FrameStack(env, stacked_frames)

        super(DeepMindAtari2600Wrapper, self).__init__(env)

        self._compress_frames = not compress_frames

    def reset(self):
        observation = super().reset()
        return np.array(observation) if self._compress_frames else observation

    def step(self, action):
        observation, reward, done, info = super().step(action)
        observation = np.array(observation) if self._compress_frames else observation
        return observation, reward, done, info
