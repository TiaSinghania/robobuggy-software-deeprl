import gymnasium as gym
import argparse
import tempfile
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.util.util import make_vec_env
from imitation.policies.serialize import load_policy
from imitation.algorithms.dagger import reconstruct_trainer
from imitation.util.logger import configure
from src.simulator.environment import BuggyCourseEnv
from src.policy_wrappers.policy_wrapper import PolicyWrapper

import matplotlib.pyplot as plt
import pandas as pd


class DAgger_Wrapper(PolicyWrapper):

    def __init__(self, **kwargs):
        print(kwargs)
        super().__init__(**kwargs)

        rng = np.random.default_rng(0)
        self.env = make_vec_env(self.env.unwrapped.spec.id, rng=rng, n_envs=1)
        self.log_path = self.dirpath + "/dagger"
        self.logger = configure(self.log_path, ('log', 'csv'))
        expert = load_policy(
            "stanley-policy",
            venv=self.env,
            reference_traj_path="src/util/buggycourse_sc.json",
        )
        bc_trainer = bc.BC(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            rng=rng,
        )
        self.dagger_trainer = SimpleDAggerTrainer(
            venv=self.env,
            scratch_dir=self.log_path,
            expert_policy=expert,
            bc_trainer=bc_trainer,
            rng=rng,
            custom_logger=self.logger
        )
        self.policy = self.dagger_trainer.policy


    def train(self, timesteps):
        print("Training DAgger model...")
        self.dagger_trainer.train(timesteps)
        df = pd.read_csv(f"{self.log_path}/progress.csv")
        print(df.columns)
        df.plot(y="loss")
        plt.savefig(self.dirpath + "/dagger_loss.png")
        plt.show()

        print("Training complete. ")

    def save(
        self,
    ):
        self.dagger_trainer.save_trainer()
        print("Model saved to dagger-buggy-course")

    def load(self):
        print("Loading existing DAgger model...")

        self.dagger_trainer = reconstruct_trainer(
            scratch_dir=self.log_path, venv=self.env
        )
        self.policy = self.dagger_trainer.policy
