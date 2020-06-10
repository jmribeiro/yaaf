import argparse

from examples.pursuit import Pursuit
from examples.pursuit.metrics import PreyCapturesEveryTimestepIntervalMetric

from yaaf.agents.dqn import MLPDQNAgent as DQNAgent
from yaaf.execution import TimestepRunner
from yaaf.visualization import LinePlot

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-timesteps', type=int, default=5000)
    parser.add_argument('-eval_interval', type=int, default=500)
    parser.add_argument('-log_interval', type=int, default=500)
    parser.add_argument('-render', action="store_true")
    parser.add_argument('-cuda', action="store_true")

    opt = parser.parse_args()

    # ##### #
    # Train #
    # ##### #

    metric = PreyCapturesEveryTimestepIntervalMetric(opt.eval_interval, verbose=True, log_interval=opt.log_interval)

    env = Pursuit(teammates="teammate aware", features="relative agent")
    dqn = DQNAgent(env.num_features, env.num_actions, cuda=opt.cuda)
    dqn.train()     # By default it's already in training mode (like torch modules)

    trainer = TimestepRunner(timesteps=opt.timesteps, agent=dqn, environment=env, observers=[metric], render=opt.render)
    trainer.run()

    plot = LinePlot(title="DQN Learning Pursuit",
                    x_label="Timesteps", y_label=metric.name,
                    num_measurements=int(opt.timesteps / opt.eval_interval),
                    x_tick_step=opt.eval_interval)
    plot.add_run("DQN", metric.result(), color="r")
    metric.reset()
    plot.show()
