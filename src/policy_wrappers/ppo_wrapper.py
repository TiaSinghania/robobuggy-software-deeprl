import gymnasium as gym
import argparse

from src.simulator.environment import BuggyCourseEnv
from stable_baselines3 import PPO
from src.policy_wrappers.policy_wrapper import PolicyWrapper


import gymnasium as gym
import argparse
import matplotlib.pyplot as plt

from src.simulator.environment import BuggyCourseEnv
from stable_baselines3 import PPO
from scripts.policies.policy import PolicyWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import plot_results
from stable_baselines3.common import results_plotter


class PPO_Wrapper(PolicyWrapper):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.policy: PPO = PPO("MlpPolicy", self.env, verbose=1)
        self.env = Monitor(self.env, self.dirpath)

    def train(self, timesteps):
        # we have something called dirpath
        print("Training PPO model...")
        self.policy.learn(total_timesteps=timesteps)
        print("Training complete.")

        plot_results([self.dirpath], timesteps, results_plotter.X_TIMESTEPS, "PPO Buggy")
        plt.savefig(self.dirpath + "/ppo_rewards.png")
        plt.show()
        return self.model

    def save(self):
        self.policy.save(f"{self.dirpath}/model")
        print("Model saved to ppo_buggy-course.")

    def load(self):
        print("Loading existing PPO model...")
        self.policy = PPO.load(f"{self.dirpath}/model")
