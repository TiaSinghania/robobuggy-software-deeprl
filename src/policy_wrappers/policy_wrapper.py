from abc import ABC, abstractmethod


class PolicyWrapper(ABC):
    def __init__(self, env, dirpath):
        self.env = env
        self.dirpath = dirpath
        self.policy = None # CHILD ATTRIBUTE

    @abstractmethod
    def train(timesteps):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass
