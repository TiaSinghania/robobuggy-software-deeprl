import gymnasium as gym
import argparse
import numpy as np

from scripts.visualize import visualize_environment
from scripts.policies.stanley_policy import stanley_policy_loader
from scripts.policies.policy import Policy


class ExpertStanley(Policy):
    def __init__(self, env, reference_traj):
        self.env = env
        self.reference_traj = reference_traj

    def train(self):
        self.model = stanley_policy_loader(self.env, reference_traj_path=self.reference_traj)
        return self.model
