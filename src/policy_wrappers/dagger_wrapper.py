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
from src.simulator.environment import BuggyCourseEnv
import src.policies.stanley_policy
from src.policy_wrappers.policy_wrapper import PolicyWrapper


class DAgger_Wrapper(PolicyWrapper):

    def __init__(self, reference_traj_path, **kwargs):
        super().__init__(**kwargs)

        rng = np.random.default_rng(0)
        self.env = make_vec_env(self.env.unwrapped.spec.id, rng=rng, n_envs=1)
        self.log_path = self.dirpath + "/dagger"
        expert = load_policy(
            "stanley-policy",
            venv=self.env,
            reference_traj_path=reference_traj_path,
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
        )
        self.policy = self.dagger_trainer.policy

    def train(self, timesteps):
        print("Training DAgger model...")
        self.dagger_trainer.train(timesteps)
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
