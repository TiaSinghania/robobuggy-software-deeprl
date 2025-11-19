import numpy as np


class RandomPolicy:
    def predict(self, obs):
        return np.random.uniform(low=-0.3, high=0.3, size=(1,)), None
