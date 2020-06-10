import math
import time
from multiprocessing import Value, RLock, Process
from typing import Sequence, Optional

from gym import Env

from yaaf.agents import Agent
from yaaf.execution import Runner


class AsynchronousParallelRunner(Process):

    def __init__(self,
                 agents: Sequence[Agent],
                 environments: Sequence[Env],
                 shared_observers: Optional[Sequence[callable]] = None,
                 max_timesteps: int = math.inf,
                 max_episodes: int = math.inf,
                 max_seconds: float = math.inf,
                 render_ids=None):

        super(AsynchronousParallelRunner, self).__init__()

        self._running = Value("b", False)

        self._num_processes = len(agents)
        self._agents = agents
        self._environments = environments

        self._max_timesteps = max_timesteps
        self._max_episodes = max_episodes
        self._max_seconds = max_seconds

        self._total_steps = Value("i", 0)
        self._total_episodes = Value("i", 0)

        self._start_time = Value("f", 0)

        def increment(counter, value):
            counter.value += value

        steps_counter = lambda timestep: increment(self._total_steps, 1)
        episodes_counter = lambda timestep: increment(self._total_episodes, timestep.is_terminal)

        render_ids = render_ids or []
        render_lock = RLock() if len(render_ids) > 1 else None

        shared_observers = shared_observers or []

        self._runners = [
            Runner(
                agent=agents[i], environment=environments[i],
                observers=shared_observers + [steps_counter, episodes_counter],
                render=i in render_ids, render_lock=render_lock
            ) for i in range(self._num_processes)]

    def start(self):
        # Processed need to be created in main thread
        self._running.value = True
        [runner.run_asynchronously(1.0) for runner in self._runners]
        super().start()

    def run(self):
        while self.running:
            done = self._check_progress()
            self._running.value = not done
        self.stop()

    def _check_progress(self):
        total_seconds = time.time() - self._start_time.value
        done = \
            self._total_steps.value >= self._max_timesteps or \
            self._total_episodes.value >= self._max_episodes or \
            total_seconds >= self._max_seconds
        return done

    def stop(self):
        [runner.stop(kill=True, kill_delay=0.0) for runner in self._runners]

    @property
    def running(self):
        return self._running.value

    @property
    def max_episodes(self):
        return self._max_episodes
