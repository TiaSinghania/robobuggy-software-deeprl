import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from src.simulator.environment import BuggyCourseEnv

# Parallel environments
vec_env = make_vec_env("BuggyCourseEnv-v1", n_envs=4)

model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=25000)
model.save("ppo_buggy-course")

del model  # remove to demonstrate saving and loading

model = PPO.load("ppo_buggy-course")

obs = vec_env.reset()
dones = [False]
while not any(dones):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
