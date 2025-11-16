import gymnasium as gym
from src.simulator.environment import BuggyCourseEnv

from stable_baselines3 import PPO

from scripts.visualize import visualize_environment

env = gym.make("BuggyCourseEnv-v1")

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=250000)
model.save("ppo_buggy-course")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_buggy-course")

visualize_environment(policy=model)

    