from gym.wrappers import LazyFrames
import numpy as np
from yaaf import Timestep
from yaaf.memory import ExperienceReplayBuffer


class LazyFramesExperienceReplayBuffer(ExperienceReplayBuffer):

    def __init__(self, max_size, sample_size, lz4_compress=False):
        super().__init__(max_size, sample_size)
        self._lz4_compress = lz4_compress

    def push(self, timestep):
        obs, action, reward, next_obs, is_terminal, info = timestep
        obs = LazyFrames(obs)
        next_obs = LazyFrames(next_obs, lz4_compress=self._lz4_compress)
        timestep = Timestep(obs, action, reward, next_obs, is_terminal, info)
        super().push(timestep)

    def sample(self):
        batch = super().sample()
        sample = []
        for timestep in batch:
            obs, action, reward, next_obs, is_terminal, info = timestep
            obs = np.array(obs)
            next_obs = np.array(next_obs)
            timestep = Timestep(obs, action, reward, next_obs, is_terminal, info)
            sample.append(timestep)
        return sample

    @property
    def all(self):
        buffer = list(self._buffer)
        all = []
        for timestep in buffer:
            obs, action, reward, next_obs, is_terminal, info = timestep
            obs = np.array(obs)
            next_obs = np.array(next_obs)
            timestep = Timestep(obs, action, reward, next_obs, is_terminal, info)
            all.append(timestep)
        return all
