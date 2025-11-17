import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm

sys.path.append("scripts")

from src.simulator.environment import BuggyCourseEnv


def visualize_environment(policy: BaseAlgorithm, render_every_n_steps=5):
    """Run the buggy environment with visualization using env.render()."""
    env = BuggyCourseEnv(rate=100, render_every_n_steps=render_every_n_steps)

    obs, _ = env.reset()
    terminated = False

    print("Starting simulation...")
    print("Press Ctrl+C to stop")

    try:
        step = 0
        while not terminated:
            # Random action for demonstration (replace with policy later)
            action, _states = policy.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)

            # Render the environment with step counter
            env.render()

            step += 1

            # Check if user closed the window
            if env.window_closed:
                print(f"\nWindow closed by user at step {step}")
                print(f"Total time: {step * env.dt:.2f}s")
                break

            if terminated:
                print(f"\nSimulation finished at step {step}!")
                print(f"Total time: {step * env.dt:.2f}s")
                # Keep the final plot open
                plt.ioff()
                plt.show()
                break

    except KeyboardInterrupt:
        print(f"\nSimulation stopped at step {step}")
        print(f"Total time: {step * env.dt:.2f}s")
        if not env.window_closed:
            plt.ioff()
            plt.show()
    finally:
        # Clean up matplotlib resources
        if env.window_closed:
            plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Buggy environment runner")
    parser.add_argument(
        "--visualize", action="store_true", help="Visualize the environment"
    )
    parser.add_argument(
        "--render-every",
        type=int,
        default=5,
        help="Render visualization every N steps (default: 5). Lower = smoother but slower.",
    )
    args = parser.parse_args()

    if args.visualize:
        visualize_environment(render_every_n_steps=args.render_every)
    else:
        print("Run with --visualize flag to see the environment")
