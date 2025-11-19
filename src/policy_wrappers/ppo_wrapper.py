import gymnasium as gym
import argparse

from src.simulator.environment import BuggyCourseEnv
from stable_baselines3 import PPO
from src.policy_wrappers.policy_wrapper import PolicyWrapper


class PPO_Wrapper(PolicyWrapper):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.policy: PPO = PPO("MlpPolicy", self.env, verbose=1)

    def train(self, timesteps):
        print("Training PPO model...")
        self.policy.learn(total_timesteps=timesteps)
        print("Training complete.")

    def save(self):
        self.policy.save(f"{self.dirpath}/model")
        print("Model saved to ppo_buggy-course.")

    def load(self):
        print("Loading existing PPO model...")
        self.policy = PPO.load(f"{self.dirpath}/model")
