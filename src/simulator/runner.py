import matplotlib.pyplot as plt
import numpy as np
import torch
import gym
import envs
import os
import logging

log = logging.getLogger("root")
log.setLevel("INFO")

INFO = 10

# Training params
TASK_HORIZON = 40
PLAN_HORIZON = 5

# Model params
LR = 1e-3

# Dims
STATE_DIM = 8


class BuggyRunner(object):
    def __init__(self, env_name="BuggyCourseEnv-v1"):
        self.env = gym.make(env_name)
        self.task_horizon = TASK_HORIZON

    def train(self, num_train_itrs, num_episodes_per_itr):
        losses = []
        accs = []
        for _ in range(num_train_itrs):
            for _ in range(num_episodes_per_itr):
                done = False
                while not done:
                    self.env.step()


    def test(self, num_episodes):
        # return average return and average success
        pass




