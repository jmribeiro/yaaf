import random
from collections import deque


class ExperienceReplayBuffer:

    def __init__(self, max_size, sample_size):
        self._buffer = deque(maxlen=max_size)
        self._sample_size = sample_size
        self._cursor = 0
        self._max_size = max_size

    def push(self, timestep):
        self._buffer.append(timestep)
        self._cursor = min(self._cursor + 1, self._max_size)

    def sample(self):
        size = min(self._sample_size, len(self))
        return random.sample(self._buffer, size)

    @property
    def all(self):
        return list(self._buffer)

    def __len__(self):
        return self._cursor


