import math
import time
from multiprocessing import Value, RLock
from typing import Sequence, Optional

from gym import Env

from yaaf.agents import Agent
from yaaf.execution import Runner


class ParallelRunner:

    def __init__(self,
                 agents: Sequence[Agent],
                 environments: Sequence[Env],
                 shared_observers: Optional[Sequence[callable]] = None,
                 max_timesteps: int = math.inf,
                 max_episodes: int = math.inf,
                 max_seconds: float = math.inf,
                 render_ids=None):

        super(ParallelRunner, self).__init__()

        self._num_processes = len(agents)
        self._agents = agents
        self._environments = environments

        self._max_timesteps = max_timesteps
        self._max_episodes = max_episodes
        self._max_seconds = max_seconds

        self._render = render_ids or []
        self._render_lock = RLock() if len(self._render) > 1 else None
        self._shared_observers = shared_observers or []

        self._running = Value("b", False)

        self._remaining_timesteps = Value("f", max_timesteps)
        self._remaining_episodes = Value("f", max_episodes)
        self._step_lock = RLock()

        self._start_time = Value("f", 0)

        self._runners = [
            _SlaveRunner(
                agent=agents[i], environment=environments[i], observers=shared_observers,
                step_lock=self._step_lock,
                remaining_timesteps=self._remaining_timesteps, remaining_episodes=self._remaining_episodes,
                render=i in self._render, render_lock=self._render_lock
            ) for i in range(self._num_processes)]

    def run(self):
        self._running.value = True
        [runner.run_asynchronously(1.0) for runner in self._runners]
        while self.running:
            done = self._check_progress()
            self._running.value = not done
        time.sleep(1.0)
        [runner.stop(kill=True, kill_delay=0.0) for runner in self._runners]

    def _check_progress(self):
        total_seconds = time.time() - self._start_time.value
        slaves_status = [runner.running for runner in self._runners]
        done = total_seconds >= self._max_seconds or True not in slaves_status
        return done

    @property
    def running(self):
        return self._running.value

    @property
    def max_episodes(self):
        return self._max_episodes

    @property
    def total_episodes(self):
        with self._step_lock:
            return self._max_episodes - self._remaining_episodes.value

    def reset(self):

        self._running = Value("b", False)

        self._remaining_timesteps = Value("f", self._max_timesteps)
        self._remaining_episodes = Value("f", self._max_episodes)
        self._step_lock = RLock()

        self._start_time = Value("f", 0)

        self._runners = [
            _SlaveRunner(
                agent=self._agents[i], environment=self._environments[i], observers=self._shared_observers,
                step_lock=self._step_lock,
                remaining_timesteps=self._remaining_timesteps, remaining_episodes=self._remaining_episodes,
                render=i in self._render, render_lock=self._render_lock
            ) for i in range(self._num_processes)]


class _SlaveRunner(Runner):

    def __init__(self, agent, environment, observers, step_lock, remaining_timesteps, remaining_episodes, render, render_lock):
        super(_SlaveRunner, self).__init__(agent, environment, observers, render=render, render_lock=render_lock)
        self._remaining_timesteps = remaining_timesteps
        self._remaining_episodes = remaining_episodes
        self._step_lock = step_lock

    def run(self):

        time.sleep(self._start_delay)
        self._running.value = True
        self._start_time = time.time()

        self.reset()
        self._environment.reset()
        self._render_step()

        while self._can_step():
            with self._step_lock:
                if self._remaining_timesteps.value == 0 or self._remaining_episodes.value == 0:
                    break
                else:
                    timestep = self.step()
                    self._remaining_timesteps.value -= 1
                    self._remaining_episodes.value -= timestep.is_terminal

        self._environment.close()
        self._running.value = False
