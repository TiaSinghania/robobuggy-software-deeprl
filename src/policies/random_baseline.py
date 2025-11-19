import gymnasium as gym
import numpy as np
import argparse

from scripts.policies.policy import Policy

class Random(Policy):
    class RandomPolicy:
        def predict(self, obs):
            return np.random.uniform(low=-0.3, high=0.3, size=(1,)), None

    def train(self):
        return Random.RandomPolicy()
