# Yet Another Agents Framework

_An RL research-oriented framework for agent prototyping and evaluation_

- [Introduction](#introduction)
- [Installation](#installation)
- [Examples](#examples)
    - [DQN for Space Invaders](#1---space-invaders-dqn)
    - [DQN for Cart Pole](#2---cartpole-dqn)
    - [A3C (on GPU) for Space Invaders](#3---asynchronous-advantage-actor-critic-on-gpu)
    - [DQN from scratch for CartPole](#4---cartpole-dqn-from-scratch)
- [Markov Submodule](#markov)    
- [Citing](#citing-the-project)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)

## Introduction

YAAF is a reinforcement learning, research-oriented framework designed for quick agent prototyping and evaluation.

At its core, YAAF follows the assumptions that:
- Agents execute actions upon the environment (which yields observations and rewards in return)
- Environments follow the interface from [OpenAI Gym](https://github.com/openai/gym/blob/master/gym/core.py)
- Agents follow a clear [agent interface](yaaf/agents/Agent.py)
- Any deep learning framework can be used for deep rl agents (even though it comes packed with [PyTorch tools](yaaf/models))

As a simple example, suppose you'd want to evaluate an agent following a random policy on the Space Invaders environment

```python
import gym

from yaaf.agents import RandomAgent
from yaaf.evaluation import AverageEpisodeReturnMetric
from yaaf.execution import EpisodeRunner
from yaaf.visualization import LinePlot

env = gym.make("SpaceInvaders-v0")
agent = RandomAgent(num_actions=env.action_space.n)
metric = AverageEpisodeReturnMetric()
runner = EpisodeRunner(5, agent, env, [metric], render=True).run()

plot = LinePlot("Space Invaders Random Policy", x_label="Episode", y_label="Average Episode Return", num_measurements=5)
plot.add_run("random policy", metric.result())
plot.show()

```

_Quick Disclaimer:_

YAAF is not yet another deep reinforcement learning framework. 

If you are looking for high-quality implementations of state-of-the-art algorithms, then I suggest the following libraries:

- [OpenAI Spinning Up (2020)](https://spinningup.openai.com/en/latest/)
- [Acme (2020)](https://github.com/deepmind/acme)
- [Stable-Baselines (2019)](https://github.com/hill-a/stable-baselines)
- [RLLib (2019)](https://github.com/ray-project/ray/tree/master/rllib)

## Installation
For the first installation I suggest setting up new Python 3.7 virtual environment
    
    $ python -m venv yaaf_test_environment
    $ source yaaf_test_environment/bin/activate
    $ pip install --upgrade pip setuptools
    $ pip install yaaf  
    $ pip install gym[atari] # Optional - Atari2600

## Examples

##### 1 - Space Invaders DQN

```python
import gym
from yaaf.environments.wrappers import DeepMindAtari2600Wrapper
from yaaf.agents.dqn import DeepMindAtariDQNAgent
from yaaf.execution import TimestepRunner
from yaaf.evaluation import AverageEpisodeReturnMetric, TotalTimestepsMetric

env = DeepMindAtari2600Wrapper(gym.make("SpaceInvaders-v0"))
agent = DeepMindAtariDQNAgent(num_actions=env.action_space.n)

metrics = [AverageEpisodeReturnMetric(), TotalTimestepsMetric()]
runner = TimestepRunner(1e9, agent, env, metrics, render=True).run()
```

##### 2 - CartPole DQN

```python
import gym
from yaaf.agents.dqn import MLPDQNAgent
from yaaf.execution import EpisodeRunner
from yaaf.evaluation import AverageEpisodeReturnMetric, TotalTimestepsMetric

env = gym.make("CartPole-v0")
layers = [(64, "relu"), (64, "relu")]
agent = MLPDQNAgent(num_features=env.observation_space.shape[0], num_actions=env.action_space.n, layers=layers)

metrics = [AverageEpisodeReturnMetric(), TotalTimestepsMetric()]
runner = EpisodeRunner(100, agent, env, metrics, render=True).run()
```

##### 3 - Asynchronous Advantage Actor-Critic on GPU 
##### (my multi-task implementation, requires tensorflow-gpu)
https://research.nvidia.com/publication/reinforcement-learning-through-asynchronous-advantage-actor-critic-gpu

```python
from yaaf.environments.wrappers import NvidiaAtari2600Wrapper
from yaaf.agents.hga3c import HybridGA3CAgent
from yaaf.execution import AsynchronousParallelRunner

num_processes = 8

envs = [NvidiaAtari2600Wrapper("SpaceInvadersDeterministic-v4")
        for _ in range(num_processes)
]

hga3c = HybridGA3CAgent(
            environment_names=[env.spec.id for env in envs],
            environment_actions=[env.action_space.n for env in envs],
            observation_space=envs[0].observation_space.shape
        )

hga3c.start_threads()
trainer = AsynchronousParallelRunner(
    agents=hga3c.workers,
    environments=envs,
    max_episodes=150000,
    render_ids=[0, 1, 2]
)

trainer.start()
while trainer.running:
    continue

hga3c.save(f"hga3c_space_invaders")
hga3c.stop_threads()
```

##### 4 - CartPole DQN from scratch

```python
import gym
from yaaf.agents.dqn import DQNAgent
from yaaf.agents.dqn.networks import DeepQNetwork

from yaaf.execution import EpisodeRunner
from yaaf.evaluation import AverageEpisodeReturnMetric, TotalTimestepsMetric
from yaaf.models.feature_extraction import MLPFeatureExtractor

# Setup env
env = gym.make("CartPole-v0")
num_features = env.observation_space.shape[0]
num_actions = env.action_space.n

# Setup model
mlp_feature_extractor = MLPFeatureExtractor(num_inputs=num_features, layers=[(64, "relu"), (64, "relu")])
network = DeepQNetwork(feature_extractors=[mlp_feature_extractor],
                       num_actions=num_actions, learning_rate=0.001, optimizer="adam", cuda=True)
# Setup agent
agent = DQNAgent(network, num_actions, 
                 discount_factor=0.95, initial_exploration_steps=1000, final_exploration_rate=0.001)

# Run
metrics = [AverageEpisodeReturnMetric(), TotalTimestepsMetric()]
runner = EpisodeRunner(100, agent, env, metrics, render=True).run()
```

## Markov Sub-module

TODO

## Citing the Project

When using YAAF in your projects, cite using:

```
@misc{yaaf,
  author = {João Ribeiro},
  title = {YAAF - Yet Another Agents Framework},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/jmribeiro/yaaf}},
}
```

## Roadmap

- Documentation
- Code cleanup
- More algorithms
    
## Contributing

If you want to contribute to this project, feel free to contact me by [e-mail](mailto:joao.mg.ribeiro94@gmail.com) or open an issue.

## Acknowledgments

YAAF was developed as a side-project to my research work and its creation was motivated by work done in the project [Ad Hoc Teams With Humans And Robots](http://gaips.inesc-id.pt/component/gaips/projects/showProject/10/44) funded by the Air Force Office of Scientific Research, in collaboration with PUC-Rio.
