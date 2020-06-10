from multiprocessing import Value
from threading import Thread
import numpy as np


class PredictorThread(Thread):

    def __init__(self, id, prediction_queue, network, worker_request_queues, batch_size):
        super(PredictorThread, self).__init__()
        self.setDaemon(True)
        self._master_prediction_queue = prediction_queue
        self._worker_request_queues = worker_request_queues
        self._network = network
        self._id = id
        self._batch_size = batch_size
        self.running = Value("b", True)

    def run(self):

        while self.running.value:

            env_batches = dict()

            for i in range(self._batch_size):

                id, env_name, observation = self._master_prediction_queue.get()

                if env_name not in env_batches:
                    ids = [id]
                    observation_requests = np.zeros((self._batch_size, *self._network.observation_space), dtype=np.float32)
                    observation_requests[0] = observation
                    env_batches[env_name] = ids, observation_requests
                else:
                    ids, observation_requests = env_batches[env_name]
                    ids.append(id)
                    observation_requests[len(ids)-1] = observation
                    env_batches[env_name] = ids, observation_requests

                if self._master_prediction_queue.empty():
                    break

            for env_name, batch in env_batches.items():

                ids, observation_requests = batch
                policies, values = self._network.predict(env_name, observation_requests[:len(ids)])

                for id in range(len(ids)):
                    worker_id = ids[id]
                    policy, value = policies[id], values[id]
                    self._worker_request_queues[worker_id].put((policy, value))
