from torch.utils.tensorboard import SummaryWriter

from yaaf import flatten_dict


class TensorboardLogger:

    def __init__(self, metrics=None, directory="tensorboard_session"):

        self._writer = SummaryWriter(directory)
        self._metrics = metrics or []
        self._step = 0
        self._episode = 0

    def __call__(self, timestep):
        self._step += 1
        self._episode += 1 if timestep.is_terminal else 0
        self._writer.add_scalar("episode", self._episode, self._step)
        [self._writer.add_scalar(metric.name, metric(timestep), self._step) for metric in self._metrics]
        info = flatten_dict(timestep.info, separator="/")
        [self._writer.add_scalar(key, value, self._step) for key, value in info.items()]

    def reset(self):
        self._step = 0
        self._episode = 0
        [metric.reset() for metric in self._metrics]
