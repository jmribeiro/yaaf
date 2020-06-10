import math
import time
from multiprocessing import Value, Process, RLock
from typing import Optional, Sequence

from gym import Env

from yaaf import Timestep
from yaaf.agents import Agent


class Runner(Process):

    def __init__(self,
                 agent: Agent,
                 environment: Env,
                 observers: Optional[Sequence[callable]] = None,
                 max_timesteps: int = math.inf,
                 max_episodes: int = math.inf,
                 max_seconds: float = math.inf,
                 render: bool = False,
                 render_lock: Optional[RLock] = None):

        super(Runner, self).__init__()

        from yaaf.environments.wrappers import ResetOnTerminalWrapper

        self._agent = agent

        self._environment = ResetOnTerminalWrapper(environment)

        self._observers = observers or []

        self._max_timesteps = max_timesteps
        self._max_episodes = max_episodes
        self._max_seconds = max_seconds

        self._total_steps = 0
        self._total_episodes = 0
        self._total_seconds = 0
        self._start_time = 0

        self._start_delay = 0.0
        self._render = render
        self._render_lock = render_lock if render else None
        self._running = Value('b', False)
        self._asynchronous = Value('b', False)

    def run(self):

        time.sleep(self._start_delay)
        self._running.value = True
        self._start_time = time.time()

        self.reset()
        self._environment.reset()
        self._render_step()

        while self._can_step():
            self.step()

        self._environment.close()
        self._running.value = False

    def reset(self):
        self._total_steps = 0
        self._total_episodes = 0
        self._total_seconds = 0.0

    def step(self):

        if self._environment.is_terminal:
            observation = self._environment.reset()
            self._render_step()
        else:
            observation = self._environment.observation

        action = self._agent.action(observation)

        next_observation, reward, is_terminal, info = self._environment.step(action)
        self._render_step()

        timestep = Timestep(observation, action, reward, next_observation, is_terminal, info)

        self._agent.reinforcement(timestep)
        [observer(timestep) for observer in self._observers]

        self._total_steps += 1
        self._total_episodes += timestep.is_terminal
        self._total_seconds = time.time() - self._start_time

        return timestep

    def episode(self):
        is_terminal = False
        trajectory = [self._environment.reset()]
        while not is_terminal:
            timestep = self.step()
            trajectory.append(timestep)
        return trajectory

    def _can_step(self):

        if self._running.value:

            # Check completion
            self._total_seconds = time.time() - self._start_time
            done = \
                self.total_steps >= self._max_timesteps or \
                self.total_episodes >= self._max_episodes or \
                self.total_seconds >= self._max_seconds

            return not done

        else:
            return False

    def start(self, start_delay=0.0):
        self._start_delay = start_delay
        self._running.value = True
        super().start()

    def run_asynchronously(self, start_delay=0.0):
        self._asynchronous.value = True
        self.start(start_delay)

    def stop(self, kill=False, kill_delay=0.0):
        # Check Join
        self._running.value = False
        if self._asynchronous and kill:
            time.sleep(kill_delay)
            self.kill()

    def _render_step(self):
        if self._render and self._render_lock is not None:
            return self._environment.render(lock=self._render_lock)
        elif self._render:
            return self._environment.render()
        else:
            return None

    # ########## #
    # Properties #
    # ########## #

    @property
    def running(self):
        return self._running.value

    @property
    def total_steps(self):
        return self._total_steps

    @property
    def total_episodes(self):
        return self._total_episodes

    @property
    def total_seconds(self):
        return self._total_seconds

    @property
    def max_steps(self):
        return int(self._max_timesteps)

    @property
    def max_episodes(self):
        return int(self._max_episodes)

    @property
    def max_seconds(self):
        return float(self._max_seconds)


class EpisodeRunner(Runner):

    def __init__(self, episodes, agent, environment, observers=None,
                 render=False, render_lock=None):
        super(EpisodeRunner, self).__init__(agent, environment, observers,
                                            max_episodes=episodes, render=render, render_lock=render_lock)


class TimestepRunner(Runner):
    def __init__(self, timesteps, agent, environment, observers=None,
                 render=False, render_lock=None):
        super(TimestepRunner, self).__init__(agent, environment, observers,
                                             max_timesteps=timesteps, render=render, render_lock=render_lock)


class TimeLimitRunner(Runner):
    def __init__(self, seconds, agent, environment, observers=None,
                 render=False, render_lock=None):
        super(TimeLimitRunner, self).__init__(agent, environment, observers,
                                              max_seconds=seconds, render=render, render_lock=render_lock)
