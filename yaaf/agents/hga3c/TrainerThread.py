from multiprocessing import Value
from threading import Thread
import numpy as np


class TrainerThread(Thread):

    def __init__(self, id, training_timestep_counter, training_queue, network, batch_size):
        super(TrainerThread, self).__init__()
        self.setDaemon(True)
        self._id = id
        self._network = network
        self._training_queue = training_queue
        self._training_timestep_counter = training_timestep_counter
        self._batch_size = batch_size
        self.running = Value("b", True)

    def run(self):

        while self.running.value:

            batch_size = 0
            env_batches = dict()

            while batch_size <= self._batch_size:

                o, a, r, env_name = self._training_queue.get()

                if env_name not in env_batches:
                    observations = o
                    actions = a
                    rewards = r
                    env_batches[env_name] = observations, actions, rewards
                else:
                    observations, actions, rewards = env_batches[env_name]
                    observations = np.concatenate((observations, o))
                    actions = np.concatenate((actions, a))
                    rewards = np.concatenate((rewards, r))
                    env_batches[env_name] = observations, actions, rewards

                batch_size += o.shape[0]

            for environment, batch in env_batches.items():
                observations, actions, rewards = batch
                self._network.fit(environment, observations, actions, rewards)

            self._training_timestep_counter.value += batch_size
