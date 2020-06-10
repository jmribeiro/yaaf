from typing import Union

from gym import Wrapper, Env


class FireToResetWrapper(Wrapper):

    def __init__(self, env: Union[Env, Wrapper]):
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE', "Environment does not have a FIRE action."
        assert len(env.unwrapped.get_action_meanings()) >= 3, "Environment does not have more than 3 actions."
        super(FireToResetWrapper, self).__init__(env)

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        observation, _, is_terminal, _ = self.env.step(1)
        if is_terminal:
            self.env.reset(**kwargs)
        observation, _, is_terminal, _ = self.env.step(2)
        if is_terminal:
            self.env.reset(**kwargs)
        return observation

    def step(self, action):
        return self.env.step(action)
