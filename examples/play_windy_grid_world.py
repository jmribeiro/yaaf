import argparse

from yaaf.agents import HumanAgent
from yaaf.environments.mdp.WindyGridWorldMDP import WindyGridWorldMDP
from yaaf.execution import EpisodeRunner

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-name', default="Human Agent")

    opt = parser.parse_args()
    env = WindyGridWorldMDP()
    agent = HumanAgent(env.action_meanings, env.num_actions, opt.name)
    runner = EpisodeRunner(1, agent, env, render=True)
    runner.run()
