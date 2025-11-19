import gymnasium as gym
import argparse
import matplotlib.pyplot as plt

from src.simulator.environment import BuggyCourseEnv
from stable_baselines3 import PPO
from scripts.policies.policy import Policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import plot_results
from stable_baselines3.common import results_plotter

class PPO(Policy):
    def __init__(self, env, dir):
        self.env = Monitor(env, dir)
        self.dir = dir
    def train(self):
        # we have something called dirpath
        print("Training PPO model...")
        self.model = PPO("MlpPolicy", self.env, verbose=1)
        self.model.learn(total_timesteps=int(1e6))
        print("Training complete.")
        plot_results([self.dir], int(1e6), results_plotter.X_TIMESTEPS, "PPO Buggy")
        plt.savefig(self.dir + "/ppo_rewards.png")
        plt.show()
        return self.model


    def save(self, ):
        self.model.save("ppo_buggy-course")
        print("Model saved to ppo_buggy-course.")


    def load(self):
        print("Loading existing PPO model...")
        return PPO.load("ppo_buggy-course")