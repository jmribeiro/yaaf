import gym

from yaaf.agents import RandomAgent
from yaaf.evaluation import AverageEpisodeReturnMetric, TotalTimestepsMetric
from yaaf.execution import EpisodeRunner
from yaaf.visualization import LinePlot

env = gym.make("SpaceInvaders-v0")
agent = RandomAgent(num_actions=env.action_space.n)
metrics = [AverageEpisodeReturnMetric(), TotalTimestepsMetric()]
runner = EpisodeRunner(5, agent, env, metrics, render=True).run()

plot = LinePlot("Space Invaders Random Policy", x_label="Episode", y_label="Average Episode Return", num_measurements=5, x_tick_step=1)
plot.add_run("random policy", metrics[0].result())
plot.show()
