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
import scripts.policies.stanley_policy
from scripts.policies.policy import Policy


class DAggerStanley(Policy):
    def train(self):
        rng = np.random.default_rng(0)
        self.env = make_vec_env(self.env.unwrapped.spec.id, rng=rng, n_envs=1)
        expert = load_policy(
            "stanley-policy",
            venv=self.env,
            reference_traj_path="src/util/buggycourse_sc.json",
        )
        print("Training DAgger model...")
        bc_trainer = bc.BC(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            rng=rng,
        )
        self.dagger_trainer = SimpleDAggerTrainer(
            venv=self.env,
            scratch_dir="./scratch_dagger",
            expert_policy=expert,
            bc_trainer=bc_trainer,
            rng=rng,
        )
        self.dagger_trainer.train(int(1e6))
        self.model = self.dagger_trainer.policy
        print("Training complete. ")
        return self.model


    def save(self, ):
        self.dagger_trainer.save_trainer()
        print("Model saved to dagger-buggy-course")


    def load(self):
        print("Loading existing DAgger model...")

        self.dagger_trainer = reconstruct_trainer(
            scratch_dir="./scratch_dagger", venv=self.env
        )
        model = self.dagger_trainer.policy
