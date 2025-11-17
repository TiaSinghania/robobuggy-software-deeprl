import gymnasium as gym
import argparse
import numpy as np

from scripts.visualize import visualize_environment


class RandomPolicy:
    def predict(self, obs):
        return np.random.uniform(low=-0.3, high=0.3, size=(1,)), None


def main():

    visualize_environment(policy=RandomPolicy())


if __name__ == "__main__":
    main()
