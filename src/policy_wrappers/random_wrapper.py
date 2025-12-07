import gymnasium as gym
import numpy as np
import argparse

from src.policy_wrappers.policy_wrapper import PolicyWrapper
from src.policies.random_policy import RandomPolicy


class Random_Wrapper(PolicyWrapper):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.policy = RandomPolicy()

    def train(self, timesteps: int, **kwargs) -> None:
        print("[WARNING]: Random Policy Does Not Train")
        return

    def save(self) -> None:
        pass

    def load(self, **kwargs) -> None:
        pass
