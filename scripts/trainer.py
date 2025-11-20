import os
import gymnasium as gym
import argparse
import datetime
import os

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from src.simulator.environment import BuggyCourseEnv
from scripts.visualize import visualize_environment
from src.policy_wrappers.ppo_wrapper import PPO_Wrapper
from src.policy_wrappers.random_wrapper import Random_Wrapper
from src.policy_wrappers.stanley_wrapper import Stanley_Wrapper
from src.policy_wrappers.dagger_wrapper import DAgger_Wrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy", type=str, default="random", help="What policy to run"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help=f"Train a new model instead of loading an existing one.",
    )
    parser.add_argument(
        "--timesteps",
        "-t",
        type=int,
        default=int(1e3),
        help="Number of timesteps to train the model for",
    )
    parser.add_argument(
        "--dirname",
        "-d",
        type=str,
        default="buggy-sim",
        help="Directory name to save model visualization",
    )
    args = parser.parse_args()

    # ensure logs directory exists
    os.makedirs("./logs", exist_ok=True)

    if args.train:
        now = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        dirpath = f"./logs/{now}-{args.dirname}-{args.policy}-{args.timesteps}"
    else:
        dirpath = f"./logs/{args.dirname}"
    env = gym.make("BuggyCourseEnv-v1", rate=20, max_episode_steps=4000)
    policy_wrapper = None
    match args.policy:
        case "random":
            policy_wrapper = Random_Wrapper(env=env, dirpath=dirpath)
        case "ppo":
            policy_wrapper = PPO_Wrapper(env=env, dirpath=dirpath)
        case "expert":
            policy_wrapper = Stanley_Wrapper(
                env=env,
                reference_traj="src/util/buggycourse_safe.json",
                dirpath=dirpath,
            )
        case "dagger":
            policy_wrapper = DAgger_Wrapper(
                env=env,
                dirpath=dirpath,
                reference_traj_path="src/util/buggycourse_safe.json",
            )
        case _:
            raise Exception("INVALID POLICY")

    if args.train:
        policy_wrapper.train(args.timesteps)
        policy_wrapper.save()

    else:
        policy_wrapper.load()

    visualize_environment(
        policy=policy_wrapper.policy, dir=dirpath, render_every_n_steps=25
    )


if __name__ == "__main__":
    main()
