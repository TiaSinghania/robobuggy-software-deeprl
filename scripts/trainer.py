import gymnasium as gym
import argparse

from src.simulator.environment import BuggyCourseEnv
from scripts.visualize import visualize_environment
from src.policies.ppo_baseline import PPO
from src.policies.random_baseline import Random
from src.policies.stanley_expert import ExpertStanley
from src.policies.student import DAggerStanley



def main(policy_name, policy_class):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        type=str,
        default="random",
        help="What policy to run"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help=f"Train a new {policy_name} model instead of loading an existing one.",
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default="buggy-sim",
        help="Filename to save model visualization",
    )
    args = parser.parse_args()


    env = gym.make("BuggyCourseEnv-v1")
    policy = None
    match args.policy:
        case "random":
            policy = Random(env)
        case "ppo":
            policy = PPO(env)
        case "expert":
            policy = ExpertStanley(env, reference_traj="src/util/buggycourse_sc.json")
        case "dagger":
            policy = DAggerStanley(env)
        case _:
            raise Exception("INVALID POLICY")


    if args.train:
        policy = policy_class.train()
        policy_class.save(f"{policy_name}_model")

    else:
        policy = policy_class.load(f"{policy_name}_model")

    visualize_environment(policy=policy, filename=args.file)


if __name__ == "__main__":
    main()
