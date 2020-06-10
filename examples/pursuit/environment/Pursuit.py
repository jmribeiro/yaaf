import time

from gym import Env
from gym.spaces import Box, Discrete

from examples.pursuit.environment import PyGameVisualizer
from yaaf.agents import RandomAgent
from examples.pursuit.environment import PursuitState
from examples.pursuit.agents import GreedyAgent, TeammateAwareAgent, ProbabilisticDestinationsAgent
from examples.pursuit.environment.utils import move, agent_directions, action_meanings

import numpy as np


class Pursuit(Env):

    def __init__(self, teammates="greedy", num_teammates=3, world_size=(5, 5),
                 features="default", deterministic=False, initial_state=None):

        super(Pursuit, self).__init__()

        self.action_space = Discrete(4)
        self.reward_range = (-np.inf, np.inf)
        self.metadata = {}

        self._num_agents = num_teammates + 1
        self._action_descriptions = action_meanings()
        self._feature_extraction_mode = features
        self.num_actions = 4
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.num_features,), dtype=np.float64)
        self.name = "Pursuit"

        self._world_size = world_size

        self._team_name = teammates
        self._teammates = self._initialize_teammates(teammates, num_teammates)
        self._pursuit_state = None
        self._first_render = True

        if deterministic and initial_state is not None:
            self._initial_state = lambda: initial_state
        elif deterministic:
            self._initial_state = PursuitState.random_state(num_teammates+1, world_size)
        else:
            self._initial_state = lambda: PursuitState.random_state(self._num_agents, self._world_size)
        self._deterministic = deterministic

    def reset(self):
        self._pursuit_state = self._initial_state()
        state = self.extract_features(self._pursuit_state)
        return state

    def step(self, action):
        teammates_actions = [teammate.action(self._pursuit_state.features()) for teammate in self._teammates]
        joint_action = [action] + teammates_actions
        next_pursuit_state, reward = self.transition(self._pursuit_state, joint_action, self._deterministic)
        next_state = self.extract_features(next_pursuit_state)
        is_terminal = next_pursuit_state.is_terminal
        self._pursuit_state = next_pursuit_state
        info = {"teammates actions": {}}
        for t in range(self.num_teammates): info["teammates actions"][str(t+1)] = teammates_actions[t]
        return next_state, reward, is_terminal, info

    def render(self, mode="human"):
        if self._first_render:
            self._visualizer = PyGameVisualizer(self.num_agents)
            self._visualizer.start(self._pursuit_state)
            self._first_render = False
        else:
            self._visualizer.update(self._pursuit_state)
        time.sleep(0.5) # FIXME - Find way to render properly

    def close(self):
        if not self._first_render: self._visualizer.end()

    # ########## #
    # Properties #
    # ########## #

    @property
    def num_features(self):
        if self._feature_extraction_mode == "default":
            return (self._num_agents + 1) * 2
        elif self._feature_extraction_mode == "relative agent" or self._feature_extraction_mode == "relative prey":
            return self._num_agents * 2
        else:
            raise ValueError(f"Invalid feature extraction mode {self._feature_extraction_mode}")

    @property
    def world_size(self):
        return self._world_size

    @property
    def num_agents(self):
        return self._num_agents

    @property
    def num_teammates(self):
        return self._num_agents - 1

    @property
    def teammates(self):
        return self._team_name

    # ################# #
    # Auxiliary Methods #
    # ################# #

    @staticmethod
    def available_feature_extraction_modes():
        return "default", "relative agent", "relative prey"

    def extract_features(self, state: PursuitState):
        if self._feature_extraction_mode == "default":
            return state.features()
        elif self._feature_extraction_mode == "relative agent":
            return state.features_relative_agent(agent_id=0)
        elif self._feature_extraction_mode == "relative prey":
            return state.features_relative_prey()
        else:
            raise ValueError(f"Invalid feature extraction mode {self._feature_extraction_mode}")

    @staticmethod
    def reward(next_pursuit_state, caught_reward=100, loose_reward=-1):
        return caught_reward if next_pursuit_state.is_terminal else loose_reward

    @staticmethod
    def transition(pursuit_state, joint_action, deterministic=False):

        action_space = agent_directions()
        world_size = pursuit_state.world_size
        num_agents = len(pursuit_state.agents_positions)
        num_preys = len(pursuit_state.prey_positions)
        occupied_positions = set(pursuit_state.prey_positions) | set(pursuit_state.agents_positions)

        directions = [action_space[a] for a in joint_action]
        agents_positions = [None] * num_agents
        prey_positions = [None] * num_preys
        agent_indices = [(i, True) for i in range(num_agents)] + [(i, False) for i in range(num_preys)]

        if not deterministic:
            np.random.shuffle(agent_indices)

        for i, is_agent in agent_indices:

            if is_agent:
                position = pursuit_state.agents_positions[i]
                direction = directions[i]
            else:
                position = pursuit_state.prey_positions[i]
                direction = PursuitState.move_prey_randomly()

            new_position = move(position, direction, world_size)

            # If collision is detected, just go to the original position
            if new_position in occupied_positions:
                new_position = position

            occupied_positions.remove(position)
            occupied_positions.add(new_position)

            if is_agent:
                agents_positions[i] = new_position
            else:
                prey_positions[i] = new_position

        next_pursuit_state = PursuitState(tuple(agents_positions), tuple(prey_positions), world_size)
        reward = 100 if next_pursuit_state.is_terminal else -1.0

        return next_pursuit_state, reward

    # ############# #
    # Team Building #
    # ############# #

    @staticmethod
    def available_teams():
        return (
            "greedy",
            "teammate aware",
            "mixed"
        )

    @staticmethod
    def available_teammates():
        return (
            "greedy",
            "teammate aware",
            "dummy/random",
            "probabilistic destinations"
        )

    def _initialize_teammates(self, team_name, num_teammates):

        if team_name == "mixed":
            assert num_teammates == 3, f"Mixed team requires only three teammates."
            teammates = ("greedy", "teammate aware", "probabilistic destinations")
        elif team_name == "greedy":
            teammates = ("greedy" for _ in range(num_teammates))
        elif team_name == "teammate aware":
            teammates = ("teammate aware" for _ in range(num_teammates))
        elif team_name == "probabilistic destinations":
            teammates = ("probabilistic destinations" for _ in range(num_teammates))
        elif team_name == "random" or team_name == "dummy":
            teammates = ("dummy" for _ in range(num_teammates))
        else:
            raise ValueError(f"Invalid team {team_name}. Available teams: {self.available_teams()}")

        return [self._spawn_teammate(teammate_type, idx + 1) for idx, teammate_type in enumerate(teammates)]

    def _spawn_teammate(self, teammate_type, teammate_index):
        if teammate_type == "dummy" or teammate_type == "random":
            teammate = RandomAgent(num_actions=4)
        elif teammate_type == "greedy":
            teammate = GreedyAgent(teammate_index, self._world_size)
        elif teammate_type == "teammate aware":
            teammate = TeammateAwareAgent(teammate_index, self._world_size)
        elif teammate_type == "probabilistic destinations":
            teammate = ProbabilisticDestinationsAgent(teammate_index, self._world_size)
        else:
            raise ValueError(f"Invalid teammate {teammate_type}. Available agents: {self.available_teammates()}")
        return teammate
