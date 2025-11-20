import argparse
import sys
import os

from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm

sys.path.append("scripts")

from src.simulator.environment import BuggyCourseEnv


def visualize_environment(policy: BaseAlgorithm, render_every_n_steps=10, dir=""):
    """Run the buggy environment with visualization using env.render()."""
    env = BuggyCourseEnv(
        rate=20, render_every_n_steps=render_every_n_steps, include_pos_in_obs=False
    )
    env.render()

    metadata = dict(title="Buggy Simulation", artist="Mehul Goel")

    writer = FFMpegWriter(fps=int(0.1 / env.dt), metadata=metadata)

    os.makedirs(dir, exist_ok=True)

    filename = f"{dir}/rollout.mp4"

    obs, _ = env.reset()
    terminated = False

    print("Starting simulation...")
    print("Press Ctrl+C to stop")

    try:
        step = 0
        with writer.saving(env.fig, filename, dpi=150):
            while not terminated:
                # Random action for demonstration (replace with policy later)
                action, _states = policy.predict(obs)
                obs, reward, terminated, truncated, _ = env.step(action)

                # Render the environment with step counter
                env.render()

                # Grab current frame from env.fig
                if step % render_every_n_steps == 0:
                    writer.grab_frame()

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


def visualize_heatmap(policy: BaseAlgorithm, n_rollouts: int, dir: str):
    """
    Run n_rollouts of the loaded policy and chart a heatmap of the paths.
    Overlay each path with a transparent opacity.
    """
    # Do not render during simulation to speed it up
    env = BuggyCourseEnv(rate=20, render_every_n_steps=0, include_pos_in_obs=False)

    paths = []

    print(f"Generating heatmap with {n_rollouts} rollouts...")

    for i in range(n_rollouts):
        obs, _ = env.reset()
        terminated = False
        current_path = []

        # Add initial position
        current_path.append((env.sc.e_utm, env.sc.n_utm))

        while not terminated:
            action, _states = policy.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)

            current_path.append((env.sc.e_utm, env.sc.n_utm))

        paths.append(np.array(current_path))
        print(f"Rollout {i+1}/{n_rollouts} complete")

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot static elements (using env's data)
    # Reference Trajectory
    traj_positions = env.target_traj.positions
    ax.plot(
        traj_positions[:, 0],
        traj_positions[:, 1],
        "k--",
        linewidth=2,
        label="Reference Trajectory",
        alpha=0.5,
    )

    # Curbs - Generate curb positions if not already generated
    # We can manually generate them using the same logic as render()
    if env.curb_positions is None:
        env.curb_positions = np.concatenate(
            [
                np.array(
                    [
                        env.left_curb.get_position_by_distance(dist)
                        for dist in np.linspace(
                            0,
                            env.left_curb.distances[-1],
                            len(env.left_curb.distances) // 10,
                        )
                    ]
                ),
                np.full(
                    (1, 2), np.nan
                ),  # Splits the line segment so both curbs aren't conjoined
                np.array(
                    [
                        env.right_curb.get_position_by_distance(dist)
                        for dist in np.linspace(
                            0,
                            env.right_curb.distances[-1],
                            len(env.right_curb.distances) // 10,
                        )
                    ]
                ),
            ],
            axis=0,
        )

    ax.plot(
        env.curb_positions[:, 0],
        env.curb_positions[:, 1],
        "k",
        linewidth=1,
        label="Curbs",
        alpha=0.5,
    )

    # Calculate alpha based on number of rollouts
    # Heuristic: More rollouts -> Less opacity per line
    # Try to keep accumulated opacity around some constant
    target_accumulated_alpha = 1.0
    alpha = min(max(target_accumulated_alpha / n_rollouts, 0.005), 0.5)

    print(f"Plotting paths with alpha={alpha:.4f}")

    # Plot all paths
    for idx, path in enumerate(paths):
        # Only label the first one to avoid legend clutter
        label = "Buggy Path" if idx == 0 else "_nolegend_"
        ax.plot(path[:, 0], path[:, 1], "r-", linewidth=1, alpha=alpha, label=label)

    ax.set_title(f"Buggy Trajectory Heatmap ({n_rollouts} Rollouts)")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.axis("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)

    os.makedirs(dir, exist_ok=True)
    save_path = os.path.join(dir, "heatmap_paths.png")
    plt.savefig(save_path, dpi=300)
    print(f"Heatmap saved to {save_path}")
    plt.close(fig)


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
