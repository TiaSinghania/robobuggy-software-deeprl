import gymnasium as gym
import argparse
import numpy as np

from scripts.visualize import visualize_environment
from src.policies.stanley_policy import stanley_policy_loader
from src.policy_wrappers.policy_wrapper import PolicyWrapper


class Stanley_Wrapper(PolicyWrapper):
    def __init__(self, reference_traj, **kwargs):
        super().__init__(**kwargs)
        self.reference_traj = reference_traj
        self.policy = stanley_policy_loader(
            self.env, reference_traj_path=self.reference_traj
        )

    def train(self, timesteps: int, **kwargs) -> None:
        print("[WARNING] Stanley Expert Does Not Train")
        return

    def load(self, **kwargs) -> None:
        pass

    def save(self) -> None:
        pass
