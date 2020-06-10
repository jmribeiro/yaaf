import numpy as np

from yaaf.policies import policy_from_action
from yaaf.agents import Agent
from multiprocessing import Queue


class WorkerAgent(Agent):

    def __init__(self, id, environment_name, num_actions, prediction_queue, training_queue, t_max, discount_factor):

        super(WorkerAgent, self).__init__(f"Hybrid GA3C Worker {id}")

        self._id = id
        self._environment_name = environment_name
        self._num_actions = num_actions
        self._master_prediction_queue = prediction_queue
        self._master_training_queue = training_queue
        self._prediction_request_queue = Queue(maxsize=1)

        self._t = 0
        self._t_max = t_max
        self._last_policy, self._last_value = None, None
        self._experiences = []

        self._discount_factor = discount_factor

    def policy(self, observation):
        self._master_prediction_queue.put((self._id, self._environment_name, observation))
        self._last_policy, self._last_value = self._prediction_request_queue.get()
        return self._last_policy if self.trainable else policy_from_action(self._last_policy.argmax(), self._num_actions)

    def _reinforce(self, timestep):
        timestep.info["HGA3C Worker Id"] = self._id
        self._experiences.append(WorkerExperience(timestep.observation, timestep.action, self._last_policy, self._last_value, timestep.reward, timestep.is_terminal))
        if timestep.is_terminal or self._t == self._t_max:
            observations, actions, rewards = self._preprocess(self._experiences)
            self._master_training_queue.put((observations, actions, rewards, self._environment_name))
            self._t = 0
            self._experiences = [self._experiences[-1]]
        self._t += 1

    def _preprocess(self, experiences):
        experiences = self._discount_rewards(experiences)
        observations = np.array([experience.observation for experience in experiences])
        actions = np.eye(self._num_actions)[np.array([exp.action for exp in experiences])].astype(np.float32)
        rewards = np.array([exp.reward for exp in experiences])
        return observations, actions, rewards

    def _discount_rewards(self, batch):
        last_datapoint = batch[-1]
        accumulator = last_datapoint.value if not last_datapoint.is_terminal else 0.0
        for t in reversed(range(0, len(batch) - 1)):
            accumulator = batch[t].reward + self._discount_factor * accumulator
            batch[t].reward = accumulator
        return batch[:-1]

    @property
    def prediction_request_queue(self):
        return self._prediction_request_queue


class WorkerExperience:

    def __init__(self, observation, action, policy, value, reward, is_terminal):
        self.observation = observation
        self.action = action
        self.policy = policy
        self.value = value
        self.reward = reward
        self.is_terminal = is_terminal
