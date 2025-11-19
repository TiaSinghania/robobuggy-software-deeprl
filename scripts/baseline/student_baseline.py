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
import scripts.stanley_expert
from scripts.visualize import visualize_environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train a new Imitation Learning model using DAgger instead of loading an existing one.",
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default="buggy-sim",
        help="Filename to save model visualization",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(0)

    vecenv = make_vec_env("BuggyCourseEnv-v1", rng=rng, n_envs=1)

    expert = load_policy(
        "stanley-policy",
        venv=vecenv,
        reference_traj_path="src/util/buggycourse_safe.json",
    )

    if args.train:
        print("Training DAgger model...")
        bc_trainer = bc.BC(
            observation_space=vecenv.observation_space,
            action_space=vecenv.action_space,
            rng=rng,
        )
        dagger_trainer = SimpleDAggerTrainer(
            venv=vecenv,
            scratch_dir="./scratch_dagger",
            expert_policy=expert,
            bc_trainer=bc_trainer,
            rng=rng,
        )
        dagger_trainer.train(int(1e6))
        model = dagger_trainer.policy
        dagger_trainer.save_trainer()
        print("Training complete. Model saved to dagger-buggy-course")

    else:
        dagger_trainer = reconstruct_trainer(
            scratch_dir="./scratch_dagger", venv=vecenv
        )
        model = dagger_trainer.policy

    visualize_environment(policy=model, filename=args.file)


if __name__ == "__main__":
    main()
