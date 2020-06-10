from abc import abstractmethod, ABC


class Metric(ABC):

    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    @abstractmethod
    def __call__(self, timestep):
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        raise NotImplementedError()

    @abstractmethod
    def result(self):
        raise NotImplementedError()
