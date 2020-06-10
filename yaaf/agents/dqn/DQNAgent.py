import copy

import torch

from yaaf.agents import Agent
from yaaf.agents.dqn.networks import FeedForwardDQN, DeepQNetwork, DeepMindAtari2600DQN
from yaaf.memory import ExperienceReplayBuffer
from yaaf.policies import lazy_epsilon_greedy_policy, linear_annealing


class DQNAgent(Agent):

    """
    Most generalized class for a DQN agent.
    See MLPDQNAgent for DQN for feature-vector based environments
    and DeepMindAtari2600DQNAgent for DQN with CNN from Nature paper.
    """

    def __init__(self, network: DeepQNetwork, num_actions: int,
                 discount_factor=0.99,
                 initial_exploration_rate=1.0, final_exploration_rate=0.1,
                 initial_exploration_steps=50000, final_exploration_step=1000000,
                 network_update_frequency=4, target_network_update_frequency=10000,
                 replay_buffer=ExperienceReplayBuffer(max_size=1000000, sample_size=32)):

        super(DQNAgent, self).__init__("DQN Agent")

        # Networks
        self._network = network
        self._network_update_frequency = network_update_frequency
        self._target_network = copy.deepcopy(self._network)
        self._target_network_update_frequency = target_network_update_frequency

        # Policy
        self._initial_exploration_rate = initial_exploration_rate
        self._final_exploration_rate = final_exploration_rate
        self._exploration_timesteps = final_exploration_step

        # Memory
        self._replay_start_size = initial_exploration_steps
        self._replay_buffer = replay_buffer

        # Misc.
        self._discount_factor = discount_factor
        self._num_actions = num_actions

    @property
    def params(self):
        p = super().params
        p["Discount Factor"] = self._discount_factor
        return p

    @property
    def network(self):
        return self._network

    def policy(self, observation):
        return lazy_epsilon_greedy_policy(
            lambda: self.q_values(observation).numpy(),
            self._num_actions,
            self.exploration_rate
        )

    def remember(self, timestep):
        self._replay_buffer.push(timestep)

    def experience_replay(self):
        replay_batch = self._replay_buffer.sample()
        X, y = self._preprocess(replay_batch)
        training_losses, training_accuracies, _ = self._network.update(X, y, epochs=1, batch_size=X.shape[0])
        should_update_target = self.total_training_timesteps % self._target_network_update_frequency == 0
        if should_update_target:
            self._target_network.load_state_dict(self._network.state_dict().copy())
        return training_losses[-1], training_accuracies[-1]

    def replay_memory(self, preprocessed=True):
        memory = self._replay_buffer.all
        return self._preprocess(memory) if preprocessed else memory

    def _reinforce(self, timestep):

        self.remember(timestep)

        info = {
            "Exploration rate": self.exploration_rate,
            "Replay Buffer Samples": len(self._replay_buffer)
        }

        should_update_network = self.total_training_timesteps >= self._replay_start_size and self.total_training_timesteps % self._network_update_frequency == 0

        if should_update_network:
            loss, accuracy = self.experience_replay()
            info["Loss"] = loss
            info["Accuracy"] = accuracy

        return info

    @property
    def exploration_rate(self):
        if self.trainable and self.total_training_timesteps < self._replay_start_size:
            exploration_rate = 1.0
        elif self.trainable:
            exploration_rate = linear_annealing(self.total_training_timesteps - self._replay_start_size,
                                                self._exploration_timesteps,
                                                self._initial_exploration_rate,
                                                self._final_exploration_rate)
        else:
            exploration_rate = 0.0
        return exploration_rate

    def q_values(self, observation, target=False):
        network = self._network if not target else self._target_network
        observation = observation.reshape(1, *self._network.input_shape)
        q_values = network.predict(observation).cpu()
        return q_values.reshape(self._num_actions) if q_values.shape != (self._num_actions,) else q_values

    def _preprocess(self, batch):

        # Auxiliary (just for clean code)
        batch_size = len(batch)
        num_features = self._network.input_shape
        gamma = self._discount_factor
        num_actions = self._num_actions
        y = torch.zeros((batch_size, num_actions))
        target_q_fn = lambda obs: self.q_values(obs, target=True)

        # Make X
        if isinstance(num_features, int):
            # For 1-D feature extraction
            X = torch.zeros((batch_size, num_features))
        elif isinstance(num_features, tuple):
            # For N-D feature extraction
            X = torch.zeros((batch_size, *num_features))
        else: raise ValueError("Invalid input shape on DQN's _prepare_batch")

        # Make y
        for t, (observation, action, reward, next_observation, terminal, _) in enumerate(batch):
            # TODO - Vectorize these computations (batched forward instead of 1-by-one)
            target_q_values = target_q_fn(observation)
            next_target_q_values = reward if terminal else reward + gamma * target_q_fn(next_observation).argmax()
            target_q_values[action] = next_target_q_values
            X[t] = torch.tensor(observation)
            y[t] = target_q_values

        return X, y

    def save(self, directory: str):
        super().save(directory)
        self._network.save(directory)

    def load(self, directory: str):
        super().load(directory)
        self._network.load(directory)


# ##### #
# Alias #
# ##### #

class MLPDQNAgent(DQNAgent):

    def __init__(self, num_features, num_actions,
                 layers=((64, "relu"), (64, "relu")), learning_rate=0.001, optimizer="adam",
                 discount_factor=0.95, initial_exploration_rate=0.50, final_exploration_rate=0.05,
                 initial_exploration_steps=0, final_exploration_step=5000,
                 replay_buffer_size=100000, replay_batch_size=32,
                 target_network_update_frequency=1, cuda=False):

        """ Standard DQN for feature-vector-based environments"""

        network = FeedForwardDQN(num_features, num_actions, layers, learning_rate, optimizer, cuda)

        super().__init__(
            network=network,
            num_actions=num_actions,
            discount_factor=discount_factor,
            initial_exploration_rate=initial_exploration_rate, final_exploration_rate=final_exploration_rate,
            initial_exploration_steps=initial_exploration_steps, final_exploration_step=final_exploration_step,
            target_network_update_frequency=target_network_update_frequency,
            replay_buffer=ExperienceReplayBuffer(replay_buffer_size, replay_batch_size))


class DeepMindAtariDQNAgent(DQNAgent):

    def __init__(self, num_actions: int):
        network = DeepMindAtari2600DQN(num_actions)
        super().__init__(network, num_actions)
