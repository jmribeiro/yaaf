from typing import Union

from gym import Wrapper, Env


class ResetOnTerminalWrapper(Wrapper):

    def __init__(self, env: Union[Env, Wrapper]):
        super(ResetOnTerminalWrapper, self).__init__(env)
        self._observation = None
        self._is_terminal = True

    @property
    def observation(self):
        """ Returns the current observation """
        return self.reset() if self._observation is None else self._observation

    @property
    def is_terminal(self):
        return self._is_terminal

    def step(self, action):
        if self._is_terminal:
            self._observation = self.reset()
        next_observation, reward, is_terminal, info = super().step(action)
        self._observation = next_observation
        self._is_terminal = is_terminal
        return next_observation, reward, is_terminal, info

    def reset(self, **kwargs):
        self._is_terminal = False
        self._observation = super().reset(**kwargs)
        return self._observation

    def render(self, mode='human', **kwargs):
        if "lock" in kwargs:
            with kwargs["lock"]:
                return super().render(mode, **kwargs)
        else:
            return super().render(mode, **kwargs)

    def close(self):
        self._is_terminal = True
        self._observation = None
        return super().close()
