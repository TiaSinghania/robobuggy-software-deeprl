from abc import ABC, abstractmethod

class Policy(ABC):
    def __init__(self, env):
        self.env = env
    @abstractmethod
    def train():
        pass
    @abstractmethod
    def save(self,file_location):
        pass
    @abstractmethod
    def load(self, file_location):
        pass


