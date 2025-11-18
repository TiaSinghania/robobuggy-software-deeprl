import gymnasium as gym
import argparse
import tempfile
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.util.util import make_vec_env
from imitation.policies.serialize import load_policy
from src.simulator.environment import BuggyCourseEnv
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
    
    env = gym.make("BuggyCourseEnv-v1")
    vecenv = make_vec_env("BuggyCourseEnv-v1", rng=rng)

    # TO CHANGE TO CHANGE
    expert = load_policy(
        "ppo-huggingface",
        organization="10703RL",
        env_name="BuggyCourseEnv-v1",
        venv=vecenv,
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
            expert_policy=expert,
            bc_trainer=bc_trainer,
            rng=rng,
        )
        dagger_trainer.train(8000)
        model = dagger_trainer.policy
        bc_trainer.save_policy("dagger-buggy-course")
        print("Training complete. Model saved to dagger-buggy-course")

    else:
        model = bc_trainer.load_policy("dagger-buggy-course", venv=vecenv)

    visualize_environment(policy=model, filename=args.file)


if __name__ == "__main__":
    main()


