import argparse

from yaaf.agents.hga3c import HybridGA3CLogger, HybridGA3CAgent
from yaaf.environments.wrappers import NvidiaAtari2600Wrapper
from yaaf.execution import AsynchronousParallelRunner


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-env', default="SpaceInvadersDeterministic-v4")
    parser.add_argument('-episodes', type=int, default=150000)
    parser.add_argument('-workers', type=int, default=16)
    parser.add_argument('-log_interval', type=int, default=1)
    parser.add_argument('-render', choices=["off", "one", "all"], default="one")

    opt = parser.parse_args()

    if opt.render == "one":
        render_ids = [0]
    elif opt.render == "all":
        render_ids = [i for i in range(opt.workers)]
    else:
        render_ids = []

    envs = [NvidiaAtari2600Wrapper(opt.env) for _ in range(opt.workers)]

    hga3c = HybridGA3CAgent(environment_names=[env.spec.id for env in envs],
                            environment_actions=[env.action_space.n for env in envs],
                            observation_space=envs[0].observation_space.shape, start_threads=True)

    hga3c_logger = HybridGA3CLogger(hga3c.num_workers, log_interval=opt.log_interval)

    trainer = AsynchronousParallelRunner(
        agents=hga3c.workers, environments=envs,
        shared_observers=[hga3c_logger], max_episodes=opt.episodes, render_ids=render_ids)

    trainer.start()
    while trainer.running:
        pass

    hga3c.save(f"{opt.env}_hga3c")

    hga3c.stop_threads()
    hga3c_logger.stop()
