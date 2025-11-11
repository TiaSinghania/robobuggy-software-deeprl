import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

from src.simulator.environment import BuggyCourseEnv

sys.path.append("scripts")


def visualize_environment(render_every_n_steps=5):
    """Run the buggy environment with visualization using env.render()."""
    env = BuggyCourseEnv(rate=100, render_every_n_steps=render_every_n_steps)

    obs, _ = env.reset()
    terminated = False

    print("Starting simulation...")
    print("Press Ctrl+C to stop")

    step = 0
    try:
        while not terminated:
            # Random action for demonstration (replace with policy later)
            action = np.array(np.random.uniform(-0.3, 0.3)).reshape(-1)
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
        "--render-every",
        type=int,
        default=5,
        help="Render visualization every N steps (default: 5). Lower = smoother but slower.",
    )
    args = parser.parse_args()

    visualize_environment(render_every_n_steps=args.render_every)
