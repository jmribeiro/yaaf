import numpy as np

from mlflow import log_metric, log_param, set_experiment, active_run, log_metrics
from yaaf import mkdir, flatten_dict


class MLFlowLogger:

    def __init__(self, params, metrics=None, experiment_name=None):

        if experiment_name is not None: set_experiment(f"{experiment_name}")
        self._step = 0
        self._episode = 0
        for param in params:
            log_param(param, params[param])
        self._metrics = metrics or []
        run = active_run()
        self._runid = run.info.run_id
        self._session = run.info.experiment_id
        self.directory = f"mlruns/{self._session}/{self._runid}"

    def __call__(self, timestep):
        self._step += 1
        self._episode += 1 if timestep.is_terminal else 0
        log_metric("timestep", self._step, step=self._step)
        log_metric("episode", self._episode, step=self._step)
        [log_metric(metric.name, metric(timestep), step=self._step) for metric in self._metrics]
        info = flatten_dict(timestep.info, separator=" ")
        log_metrics(info, step=self._step)

    def reset(self):
        self._step = 0
        self._episode = 0
        [metric.reset() for metric in self._metrics]

    def save_numpy(self):
        mkdir(f"mlruns/{self._session}/{self._runid}/numpy")
        [np.save(f"mlruns/{self._session}/{self._runid}/numpy/{metric.name.lower().replace(' ', '_')}", metric.result()) for metric in self._metrics]