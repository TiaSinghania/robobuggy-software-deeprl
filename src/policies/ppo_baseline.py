import gymnasium as gym
import argparse

from src.simulator.environment import BuggyCourseEnv
from stable_baselines3 import PPO
from scripts.policies.policy import Policy

class PPO(Policy):
    def train(self):
        print("Training PPO model...")
        self.model = PPO("MlpPolicy", self.env, verbose=1)
        self.model.learn(total_timesteps=int(1e6))
        print("Training complete.")
        return self.model

    def save(self, ):
        self.model.save("ppo_buggy-course")
        print("Model saved to ppo_buggy-course.")


    def load(self):
        print("Loading existing PPO model...")
        return PPO.load("ppo_buggy-course")