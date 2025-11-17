import gymnasium as gym
import argparse

from src.simulator.environment import BuggyCourseEnv
from stable_baselines3 import PPO
from scripts.visualize import visualize_environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true",
                        help="Train a new PPO model instead of loading an existing one.")
    args = parser.parse_args()

    env = gym.make("BuggyCourseEnv-v1")

    if args.train:
        print("Training PPO model...")
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=2500000)
        model.save("ppo_buggy-course")
        print("Training complete. Model saved to ppo_buggy-course.")
    else:
        print("Loading existing PPO model...")
        model = PPO.load("ppo_buggy-course")

    visualize_environment(policy=model)


if __name__ == "__main__":
    main()
