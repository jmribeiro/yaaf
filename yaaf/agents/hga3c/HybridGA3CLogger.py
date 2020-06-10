from multiprocessing import Queue, Value, Process
from queue import Empty


class HybridGA3CLogger(Process):

    def __init__(self, num_workers, verbose=True, log_interval=1, reward_clipped=True):

        super(HybridGA3CLogger, self).__init__()

        self._verbose = verbose
        self._log_interval = log_interval
        self._reward_clipped = reward_clipped

        self._queue = Queue(10000)

        self._scores = [0 for _ in range(num_workers)]
        self._episodes = 0
        self._timesteps = Value("i", 0)
        self._mean = 0

        self._running = Value("b", True)

        self.start()

    def __call__(self, timestep):
        self._queue.put((
            timestep.info["HGA3C Worker Id"],
            timestep.info["Unclipped reward"] if self._reward_clipped else timestep.reward,
            timestep.is_terminal
        ))

    def run(self):

        while self._running.value:

            try:
                worker_id, reward, is_terminal = self._queue.get(block=False)
            except Empty:
                continue

            self._timesteps.value += 1
            self._scores[worker_id] += reward
            if is_terminal:
                episode_return = self._scores[worker_id]
                self._episodes += 1
                self._scores[worker_id] = 0
                self._mean = self._mean + ((episode_return - self._mean) / self._episodes)
                if self._episodes % self._log_interval == 0:
                    message = f"" \
                              f"Episode {self._episodes} (Worker #{worker_id}): " \
                              f"Return {episode_return} (Avg. {round(self._mean, 3)}) " \
                              f"| Total frames: {self._timesteps.value}\n"
                    print(message, end="", flush=True)

    def stop(self):
        self._running.value = False

    @property
    def total_timesteps(self):
        return self._timesteps.value
