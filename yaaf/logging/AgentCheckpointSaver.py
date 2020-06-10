import math


class AgentCheckpointSaver:

    def __init__(self, agent, directory, episodes=math.inf, timesteps=math.inf):

        assert episodes != math.inf or timesteps != math.inf, "Checkpoint Saver needs at least one save interval method (Episodes or Timesteps)"
        self._episode_save_interval = episodes
        self._timestep_save_interval = timesteps
        self._total_episodes = 0
        self._total_timesteps = 0
        self._agent = agent
        self._save_dir = directory

    def __call__(self, timestep):
        self._total_timesteps += 1
        if timestep.is_terminal: self._total_episodes += 1
        if self._total_timesteps % self._timestep_save_interval == 0 and self._total_timesteps > 0:
            self._agent.save(f"{self._save_dir}/{self._total_timesteps} Timesteps")
        if self._total_episodes % self._episode_save_interval == 0 and self._total_episodes > 0:
            self._agent.save(f"{self._save_dir}/{self._total_episodes} Episodes")
