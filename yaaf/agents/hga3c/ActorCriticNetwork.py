import os
from logging import warning

import numpy as np
import tensorflow.python as tf
from threading import RLock


class ActorCriticNetwork:

    def __init__(self, environment_actions, observation_space, conv2d_layers, mlp_layers,
                 learning_rate=0.0003, rms_decay=0.99, rms_momentum=0.0, rms_epsilon=0.1,
                 log_noise=1e-6, entropy_beta=0.01, device="gpu:0"):

        self._log_noise = log_noise
        self._entropy_beta = entropy_beta
        self._device = device

        self._lock = RLock()
        self._actor_critics = dict()
        self._graph = tf.Graph()

        with self._graph.as_default():
            with tf.device(device):
                self._optimizer = tf.train.RMSPropOptimizer(learning_rate, rms_decay, rms_momentum, rms_epsilon)
                self._setup_input_network_head(observation_space, conv2d_layers, mlp_layers)
                self._environment_names = list(environment_actions.keys())
                self._num_actions = list(environment_actions.values())
                self._observation_space = observation_space
                self._R = tf.placeholder(tf.float32, [None], name='R')
                [self._setup_actor_critic_tail(environment, num_actions) for environment, num_actions in environment_actions.items()]
                self._tensorflow_initialization()

    # ##### #
    # Model #
    # ##### #

    def fit(self, environment_name, observations, actions, rewards):

        actor_critic = self._actor_critics[environment_name]
        minimizer = actor_critic.minimizer
        R = actor_critic.R
        actions_one_hot = actor_critic.actions_one_hot

        with self._lock:
            _, loss = self._session.run(
                [minimizer, actor_critic.loss],
                feed_dict={
                    self._input_layer: observations,
                    R: rewards,
                    actions_one_hot: actions}
            )

    def predict(self, environment_name, observations):

        actor_critic = self._actor_critics[environment_name]
        actor = actor_critic.actor
        critic = actor_critic.critic

        with self._lock:
            return self._session.run(
                [actor, critic],
                feed_dict={
                    self._input_layer: observations}
            )

    # ##### #
    # Graph #
    # ##### #

    def _setup_input_network_head(self, observation_space, conv2d_layers, mlp_layers):

        self._input_layer = input([None, *observation_space], 'input_layer')
        last_layer = self._input_layer

        if len(conv2d_layers) > 0 and len(observation_space) == 3:
            last_layer = conv2d_feature_extractor(last_layer, conv2d_layers)

        if len(mlp_layers) > 0:
            last_layer = mlp_feature_extractor(last_layer, mlp_layers)

        self._hidden = last_layer

    def _setup_actor_critic_tail(self, environment_name, num_actions):
        if environment_name in self._actor_critics:
            return
        input_layer, last_layer = self._input_layer, self._hidden
        actor_layer, _, _ = dense(last_layer, num_actions, f'{environment_name}_policy_layer')
        critic_layer, _, _ = dense(last_layer, 1, f'{environment_name}_value_layer')
        actor = tf.nn.softmax(actor_layer)
        critic = tf.squeeze(critic_layer, axis=[1])
        loss, actions_one_hot = self._setup_actor_critic_loss(actor, critic, num_actions)
        optimizer_step = self._optimizer.minimize(loss)
        self._actor_critics[environment_name] = ActorCriticTailWrapper(actor, critic, optimizer_step, loss, self._R, actions_one_hot)

    def _setup_actor_critic_loss(self, actor, critic, num_actions):

        actions_one_hot = tf.placeholder(tf.float32, [None, num_actions])

        action_probability = tf.reduce_sum(actor * actions_one_hot, axis=1)

        log_prob = tf.log(tf.maximum(action_probability, self._log_noise))
        advantage = self._R - tf.stop_gradient(critic)
        entropy = tf.reduce_sum(tf.log(tf.maximum(actor, self._log_noise)) * actor, axis=1)

        actor_loss = -(tf.reduce_sum((log_prob * advantage), axis=0) + tf.reduce_sum((-1 * self._entropy_beta * entropy), axis=0))
        critic_loss = tf.reduce_sum(tf.square(self._R - critic), axis=0)

        loss = 0.5 * critic_loss + actor_loss

        return loss, actions_one_hot

    def _tensorflow_initialization(self):
        self._session = tf.Session(
            graph=self._graph,
            config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False,
                gpu_options=tf.GPUOptions(allow_growth=True)))
        self._session.run(tf.global_variables_initializer())
        self._saver = None

    ###############
    # Persistence #
    ###############

    def save(self, directory):
        with self._lock:
            with self._graph.as_default():
                with tf.device(self._device):
                    self._save_parameters(directory)
                    self._save_known_environments(directory)

    def load(self, directory):
        with self._lock:
            with self._graph.as_default():
                with tf.device(self._device):
                    unknown_environments = self._reload_network(directory)
                    self._load_parameters(directory, unknown_environments)

    def _reload_network(self, directory):
        env_specs = self._load_known_environments(directory)
        envs_not_in_session = list(filter(lambda env: env not in self._environment_names, env_specs.keys()))
        unknown_environments = list(filter(lambda env: env not in env_specs.keys(), self._environment_names))
        [self._setup_actor_critic_tail(env, env_specs[env]) for env in envs_not_in_session]
        self._environment_names.extend(envs_not_in_session)
        return unknown_environments

    def _save_parameters(self, directory):
        variables = tf.global_variables()
        saver = tf.train.Saver({var.name: var for var in variables}, max_to_keep=0)
        saver.save(self._session, f"{directory}/params")

    def _load_parameters(self, directory, unknown_environments):
        filename = tf.train.latest_checkpoint(os.path.dirname(f"{directory}/checkpoint"))
        params = tf.global_variables()
        saved_params = tf.train.NewCheckpointReader(filename).get_variable_to_shape_map()
        params_to_load = {var.name: var for var in params}
        for parameter in params:
            if parameter.name not in saved_params.keys():
                new_env_name = [env_name for env_name in self._actor_critics if env_name in parameter.name and env_name not in unknown_environments]
                if len(new_env_name) > 0:
                    unknown_environments.append(new_env_name[0])
                del params_to_load[parameter.name]
        if len(unknown_environments) > 0:
            warning(f"No Actor-Critics for {unknown_environments} found, spawned new policy-value layers")
        saver = tf.train.Saver(params_to_load, max_to_keep=0)
        saver.restore(self._session, filename)

    def _save_known_environments(self, directory):
        import yaml
        specs = dict(zip(self._environment_names, self._num_actions))
        with open(f"{directory}/envs.yaml", 'w') as file:
            yaml.dump(specs, file, default_flow_style=False)

    @staticmethod
    def _load_known_environments(directory):
        import yaml
        with open(f"{directory}/envs.yaml", 'r') as file:
            specs = yaml.load(file, Loader=yaml.FullLoader)
        return specs

    @property
    def observation_space(self):
        return self._observation_space


class ActorCriticTailWrapper:

    def __init__(self, actor, critic, minimizer, loss, R, actions_one_hot):
        self.actor = actor
        self.critic = critic
        self.minimizer = minimizer
        self.loss = loss
        self.R = R
        self.actions_one_hot = actions_one_hot


# ################ #
# Tensorflow Stuff #
# ################ #

activations = {
    "relu": tf.nn.relu
}


def input(shape, name):
    return tf.placeholder(tf.float32, shape, name)


def dense(previous_layer, output_shape, name, activation_function=None):
    input_shape = previous_layer.get_shape().as_list()[-1]
    random_initializer = 1.0 / np.sqrt(input_shape)
    with tf.variable_scope(name):
        weight_initializer = tf.random_uniform_initializer(-random_initializer, random_initializer)
        bias_initializer = tf.random_uniform_initializer(-random_initializer, random_initializer)
        weights = tf.get_variable('w', dtype=tf.float32, shape=[input_shape, output_shape], initializer=weight_initializer)
        biases = tf.get_variable('b', shape=[output_shape], initializer=bias_initializer)
        dot_product = tf.matmul(previous_layer, weights) + biases
        output = activation_function(dot_product) if activation_function is not None else dot_product
    return output, weights, biases


def conv2d(previous_layer, filter_size, output_shape, name, stride, padding="SAME", activation_function=None):
    input_shape = previous_layer.get_shape().as_list()[-1]
    random_initializer = 1.0 / np.sqrt(filter_size * filter_size * input_shape)
    with tf.variable_scope(name):
        weight_initializer = tf.random_uniform_initializer(-random_initializer, random_initializer)
        bias_initializer = tf.random_uniform_initializer(-random_initializer, random_initializer)
        weights = tf.get_variable('w',
                                  shape=[filter_size, filter_size, input_shape, output_shape],
                                  dtype=tf.float32,
                                  initializer=weight_initializer)
        biases = tf.get_variable('b',
                                 shape=[output_shape],
                                 initializer=bias_initializer)
        convolution = tf.nn.conv2d(previous_layer, weights, strides=[1, stride, stride, 1], padding=padding) + biases
        output = activation_function(convolution) if activation_function is not None else convolution
    return output, weights, biases


def flatten(previous_layer):
    return tf.reshape(previous_layer, shape=[-1, (previous_layer.get_shape()[1] * previous_layer.get_shape()[2] * previous_layer.get_shape()[3])])


def conv2d_feature_extractor(last_layer, layer_specs):

    for i, conv2d_spec in enumerate(layer_specs):
        num_kernels, kernel_size, stride, activation = conv2d_spec
        last_layer, _, _ = conv2d(
            last_layer, kernel_size, num_kernels,
            name=f'conv{i + 1}', stride=stride, activation_function=activations[activation]
        )
        if i == len(layer_specs) - 1:
            last_layer = flatten(last_layer)
    return last_layer


def mlp_feature_extractor(last_layer, layer_specs):
    for i, dense_spec in enumerate(layer_specs):
        units, activation = dense_spec
        last_layer, _, _ = dense(last_layer, units, f'hidden{i + 1}', activation_function=activations[activation])
    return last_layer
