import time
from multiprocessing import Queue


from yaaf.agents import Agent
from yaaf.agents.hga3c.PredictorThread import PredictorThread
from yaaf.agents.hga3c.TrainerThread import TrainerThread
from yaaf.agents.hga3c.WorkerAgent import WorkerAgent
from yaaf.models.feature_extraction import Conv2dSpec, LinearSpec


class HybridGA3CAgent(Agent):

    """
        Hybrid GA3C agent class
            A multi-task version of the Asynchronous Advantage Actor-Critic on GPU (GA3C)
            where workers may have different environments
    """

    def __init__(self, environment_names, environment_actions,
                 observation_space,
                 conv2d_layers=(
                     Conv2dSpec(num_kernels=16, kernel_size=8, stride=4, activation="relu"),
                     Conv2dSpec(num_kernels=32, kernel_size=4, stride=2, activation="relu")
                 ),
                 mlp_layers=(LinearSpec(units=256, activation="relu"),),
                 num_predictor_threads=2, num_trainer_threads=2, prediction_queue_size=100, training_queue_size=100,
                 learning_rate=0.0003, rms_decay=0.99, rms_momentum=0.0, rms_epsilon=0.1, batch_size=128,
                 log_noise=1e-6, entropy_beta=0.01, device="gpu:0",
                 t_max=5, discount_factor=0.99, start_threads=False):

        assert len(environment_names) == len(environment_actions)

        super(HybridGA3CAgent, self).__init__("HybridGA3C")

        environment_actions_dict = dict(zip(environment_names, environment_actions))

        # ########### #
        # Setup Model #
        # ########### #

        from yaaf.agents.hga3c.ActorCriticNetwork import ActorCriticNetwork
        self._network = ActorCriticNetwork(
            environment_actions_dict, observation_space, conv2d_layers, mlp_layers,
            learning_rate, rms_decay, rms_momentum, rms_epsilon,
            log_noise, entropy_beta, device)

        self._prediction_queue = Queue(maxsize=prediction_queue_size)
        self._training_queue = Queue(maxsize=training_queue_size)

        # ############# #
        # Setup Workers #
        # ############# #

        self._workers = []
        for id in range(len(environment_names)):
            worker = WorkerAgent(
                id, environment_names[id], environment_actions[id],
                self._prediction_queue, self._training_queue, t_max, discount_factor)
            self._workers.append(worker)
        self._worker_prediction_request_queues = [worker.prediction_request_queue for worker in self._workers]
        self._num_predictor_threads = num_predictor_threads
        self._num_trainer_threads = num_trainer_threads
        self._predictor_threads = []
        self._trainer_threads = []
        self._batch_size = batch_size

        if start_threads:
            self.start_threads()

    def policy(self, observation):
        return self._workers[0].policy(observation)

    def _reinforce(self, timestep):
        return self._workers[0].reinforcement(timestep)

    @property
    def num_workers(self):
        return len(self._workers)

    @property
    def workers(self):
        return self._workers

    # ############################# #
    # Predictor and Trainer Threads #
    # ############################# #

    def start_threads(self):
        self.start_trainer_threads()
        self.start_predictor_threads()

    def start_predictor_threads(self):
        self.stop_predictor_threads()
        self._predictor_threads = [
            PredictorThread(id, self._prediction_queue, self._network,
                            self._worker_prediction_request_queues, self._batch_size)
            for id in range(self._num_predictor_threads)
        ]
        [thread.start() for thread in self._predictor_threads]

    def start_trainer_threads(self):
        self.stop_trainer_threads()
        self._trainer_threads = [
            TrainerThread(i, self._total_training_timesteps, self._training_queue, self._network, self._batch_size)
            for i in range(self._num_trainer_threads)
        ]
        [thread.start() for thread in self._trainer_threads]

    def stop_threads(self, timeout=0.25):
        self.stop_trainer_threads(timeout)
        self.stop_predictor_threads(timeout)

    def stop_predictor_threads(self, timeout=0.25):
        [thread.join(timeout) for thread in self._predictor_threads]
        self._predictor_threads.clear()

    def stop_trainer_threads(self, timeout=0.25):
        [thread.join(timeout) for thread in self._trainer_threads]
        self._trainer_threads.clear()

    # ########### #
    # Persistence #
    # ########### #

    def save(self, directory):
        super().save(directory)
        self._network.save(directory)

    def load(self, directory):
        super().load(directory)
        self._network.load(directory)

    def train(self):
        if not self.trainable:
            self.start_trainer_threads()
            time.sleep(1.0)
            [worker.train() for worker in self.workers]
        super().train()

    def eval(self):
        if self.trainable:
            [worker.eval() for worker in self.workers]
            time.sleep(1.0)
            self.stop_trainer_threads()
        super().eval()
