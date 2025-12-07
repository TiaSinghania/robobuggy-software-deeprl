import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
import torch

from src.simulator.environment import BuggyCourseEnv, RMAConfig
from src.policy_wrappers.rma_ppo_wrapper import RMA_PPO_Wrapper


class EvaluatorEnv(BuggyCourseEnv):
    def __init__(
        self,
        forced_mu_friction=None,
        forced_steer_offset=None,
        forced_steer_slop=None,
        forced_cornering_stiffness=None,
        forced_course_slope=None,
        **kwargs,
    ):
        self.forced_mu_friction = forced_mu_friction
        self.forced_steer_offset = forced_steer_offset
        self.forced_steer_slop = forced_steer_slop
        self.forced_cornering_stiffness = forced_cornering_stiffness
        self.forced_course_slope = forced_course_slope
        super().__init__(**kwargs)

    def set_forced_mu_friction(self, val):
        self.forced_mu_friction = val

    def set_forced_steer_offset(self, val):
        self.forced_steer_offset = val

    def set_forced_steer_slop(self, val):
        self.forced_steer_slop = val

    def set_forced_cornering_stiffness(self, val):
        self.forced_cornering_stiffness = val

    def set_forced_course_slope(self, val):
        self.forced_course_slope = val

    def _sample_domain_randomization_state(self) -> None:
        super()._sample_domain_randomization_state()
        if self.forced_mu_friction is not None:
            self.mu_friction = self.forced_mu_friction
        if self.forced_steer_offset is not None:
            self.steer_offset = self.forced_steer_offset
            self.steer_noise = lambda: np.random.normal(
                loc=self.steer_offset, scale=self.steer_slop
            )
        if self.forced_steer_slop is not None:
            self.steer_slop = self.forced_steer_slop
            self.steer_noise = lambda: np.random.normal(
                loc=self.steer_offset, scale=self.steer_slop
            )
        if self.forced_cornering_stiffness is not None:
            self.cornering_stiffness = self.forced_cornering_stiffness
        if self.forced_course_slope is not None:
            self.course_slope = self.forced_course_slope


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate domain randomization robustness."
    )
    parser.add_argument(
        "--ppo_dir", type=str, required=True, help="Path to PPO log directory"
    )
    parser.add_argument(
        "--rma_dir", type=str, required=True, help="Path to RMA log directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="domain_randomization_comparison.png",
        help="Output plot filename",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of episodes per evaluation point",
    )
    parser.add_argument(
        "--random_others",
        action="store_true",
        help="If set, other domain parameters will be randomized during evaluation. Otherwise, they are fixed to their mean values.",
    )
    args = parser.parse_args()

    # Define range for mu_friction
    mu_values = np.linspace(0.4, 1.2, 10)

    ppo_means = []
    ppo_stds = []
    rma_means = []
    rma_stds = []

    # --- Load PPO Model ---
    print(f"Loading PPO model from {args.ppo_dir}...")
    ppo_model_path = os.path.join(args.ppo_dir, "model.zip")
    if not os.path.exists(ppo_model_path):
        ppo_model_path = os.path.join(args.ppo_dir, "model")

    try:
        ppo_model = PPO.load(ppo_model_path)
        print("PPO model loaded successfully.")
    except Exception as e:
        print(f"Failed to load PPO model: {e}")
        return

    # --- Load RMA Model ---
    print(f"Loading RMA model from {args.rma_dir}...")
    from src.rma_wrappers.rma_wrapper import RMAExtractor

    include_pos_in_obs = False
    base_obs_size = 11 if include_pos_in_obs else 9
    action_size = 1
    state_action_size = base_obs_size + action_size
    env_vector_size = 5

    policy_kwargs = dict(
        features_extractor_class=RMAExtractor,
        features_extractor_kwargs=dict(
            observation_size=base_obs_size,
            state_action_size=state_action_size,
            env_vector_size=env_vector_size,
            embedding_dim=8,
            embedding_hidden_dim=128,
            adaptation_hidden_dim=128,
            adaptation_embedding_dim=8,
            phase="phase_2",
        ),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    rma_model_path = os.path.join(args.rma_dir, "model_phase2.zip")
    if not os.path.exists(rma_model_path):
        rma_model_path = os.path.join(args.rma_dir, "model_phase_2.zip")
        if not os.path.exists(rma_model_path):
            rma_model_path = os.path.join(args.rma_dir, "model.zip")

    try:
        rma_model = PPO.load(
            rma_model_path, custom_objects={"policy_kwargs": policy_kwargs}
        )
        print("RMA model loaded successfully.")
    except Exception as e:
        print(f"Failed to load RMA model: {e}")
        return

    # --- Evaluation Loop ---

    n_envs = min(args.episodes, 10)

    # Use SubprocVecEnv to utilize multiple cores, matching training setup
    # Note: EvaluatorEnv needs to be available in the worker process.
    # SubprocVecEnv uses multiprocessing, which on Linux (fork) works fine with local classes.

    def make_ppo_env():
        return EvaluatorEnv(render_every_n_steps=0, rma_config=None)

    ppo_env = make_vec_env(make_ppo_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)

    rma_config = RMAConfig(current_phase="phase_2", lookback_steps=50)

    def make_rma_env():
        return EvaluatorEnv(render_every_n_steps=0, rma_config=rma_config)

    rma_env = make_vec_env(make_rma_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)

    print(f"\nStarting evaluation with {n_envs} vectorized environments...")

    params_to_test = [
        {
            "name": "mu_friction",
            "values": np.linspace(0.4, 1.2, 10),
            "setter": "set_forced_mu_friction",
            "train_range": (0.65, 0.99),
            "label": "Friction Coefficient (mu)",
        },
        {
            "name": "steer_offset",
            "values": np.linspace(-0.1, 0.2, 10),
            "setter": "set_forced_steer_offset",
            "train_range": (0, 5 * np.pi / 180),
            "label": "Steer Offset (rad)",
        },
        {
            "name": "steer_slop",
            "values": np.linspace(0, 0.1, 10),
            "setter": "set_forced_steer_slop",
            "train_range": (0, 2 * np.pi / 180),
            "label": "Steer Slop (rad)",
        },
        {
            "name": "cornering_stiffness",
            "values": np.linspace(1000, 4500, 10),
            "setter": "set_forced_cornering_stiffness",
            "train_range": (2000, 3500),
            "label": "Cornering Stiffness (N/rad)",
        },
        {
            "name": "course_slope",
            "values": np.linspace(0, 0.1, 10),
            "setter": "set_forced_course_slope",
            "train_range": (1 * np.pi / 180, 3 * np.pi / 180),
            "label": "Course Slope (rad)",
        },
    ]

    param_means = {
        "mu_friction": np.mean((0.65, 0.99)),
        "steer_offset": np.mean((0, 5 * np.pi / 180)),
        "steer_slop": np.mean((0, 2 * np.pi / 180)),
        "cornering_stiffness": int(np.mean((2000, 3500))),
        "course_slope": np.mean((1 * np.pi / 180, 3 * np.pi / 180)),
    }

    # Map param names to their setter method names
    setter_map = {
        "mu_friction": "set_forced_mu_friction",
        "steer_offset": "set_forced_steer_offset",
        "steer_slop": "set_forced_steer_slop",
        "cornering_stiffness": "set_forced_cornering_stiffness",
        "course_slope": "set_forced_course_slope",
    }

    for param_config in params_to_test:
        param_name = param_config["name"]
        values = param_config["values"]
        setter_name = param_config["setter"]
        train_range = param_config["train_range"]
        label = param_config["label"]

        print(f"\nTesting parameter: {param_name}")

        ppo_means = []
        ppo_stds = []
        rma_means = []
        rma_stds = []

        # Reset all forced params first
        for p_name in param_means:
            ppo_env.env_method(setter_map[p_name], None)
            rma_env.env_method(setter_map[p_name], None)

        if not args.random_others:
            for p_name, p_mean in param_means.items():
                if p_name != param_name:
                    ppo_env.env_method(setter_map[p_name], p_mean)
                    rma_env.env_method(setter_map[p_name], p_mean)
            print("  Other parameters fixed to mean values.")
        else:
            print("  Other parameters randomized.")

        for val in values:
            print(f"  Evaluating {param_name} = {val:.4f}")

            ppo_env.env_method(setter_name, val)
            rma_env.env_method(setter_name, val)

            p_mean, p_std = evaluate_policy(
                ppo_model,
                ppo_env,
                n_eval_episodes=args.episodes,
                deterministic=True,
                return_episode_rewards=False,
            )
            ppo_means.append(p_mean)
            ppo_stds.append(p_std)

            r_mean, r_std = evaluate_policy(
                rma_model,
                rma_env,
                n_eval_episodes=args.episodes,
                deterministic=True,
                return_episode_rewards=False,
            )
            rma_means.append(r_mean)
            rma_stds.append(r_std)

            print(f"    PPO: Mean Reward = {p_mean:.2f} +/- {p_std:.2f}")
            print(f"    RMA: Mean Reward = {r_mean:.2f} +/- {r_std:.2f}")

        plt.figure(figsize=(10, 6))

        plt.errorbar(
            values,
            ppo_means,
            yerr=ppo_stds,
            label="PPO",
            marker="o",
            capsize=5,
            linestyle="-",
            color="blue",
        )
        plt.errorbar(
            values,
            rma_means,
            yerr=rma_stds,
            label="RMA (Phase 2)",
            marker="x",
            capsize=5,
            linestyle="-",
            color="orange",
        )

        plt.title(f"Performance vs {label}")
        plt.xlabel(label)
        plt.ylabel("Mean Reward")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)

        plt.axvspan(
            train_range[0],
            train_range[1],
            color="green",
            alpha=0.1,
            label="Training Range",
        )
        plt.legend()

        output_filename = args.output.replace(".png", f"_{param_name}.png")
        if output_filename == args.output:
            output_filename = f"{args.output}_{param_name}.png"

        base, ext = os.path.splitext(args.output)
        if not ext:
            ext = ".png"
        output_filename = f"{base}_{param_name}{ext}"

        plt.savefig(output_filename)
        print(f"Plot saved to {output_filename}")
        plt.close()


if __name__ == "__main__":
    main()
