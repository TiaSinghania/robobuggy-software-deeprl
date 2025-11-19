import gymnasium as gym
import argparse
import numpy as np

from scripts.visualize import visualize_environment
from scripts.stanley_expert import StanleyPolicy


env = gym.make("BuggyCourseEnv-v1")
model = StanleyPolicy(
    reference_traj_path="src/util/buggycourse_sc.json",
    venv=env,
)


def main():

    visualize_environment(policy=model)


if __name__ == "__main__":
    main()
