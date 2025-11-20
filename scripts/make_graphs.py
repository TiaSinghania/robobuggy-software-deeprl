import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Plot training metrics from PPO and Dagger CSV files."
    )
    parser.add_argument(
        "--ppo_file",
        type=str,
        default="data/ppo/ppo_metrics.csv",
        help="Path to PPO metrics CSV file (default: data/ppo/ppo_metrics.csv)",
    )
    parser.add_argument(
        "--dagger_file",
        type=str,
        default="data/dagger/dagger_200k.csv",
        help="Path to Dagger metrics CSV file (default: data/dagger/dagger_200k.csv)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="graphs",
        help="Directory to save the output graphs (default: graphs)",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load PPO Data ---
    ppo_data_valid = False
    ppo_df = None
    try:
        if os.path.exists(args.ppo_file):
            ppo_df = pd.read_csv(args.ppo_file)
            # Check for required columns: 'total_timesteps' and 'ep_rew_mean'
            # Note: Strip whitespace from column names just in case
            ppo_df.columns = ppo_df.columns.str.strip()

            if "total_timesteps" in ppo_df.columns and "ep_rew_mean" in ppo_df.columns:
                ppo_data_valid = True
                print(f"Successfully loaded PPO data from {args.ppo_file}")
            else:
                print(
                    f"Warning: PPO file {args.ppo_file} missing required columns ('total_timesteps', 'ep_rew_mean'). Found: {ppo_df.columns.tolist()}"
                )
        else:
            print(f"Warning: PPO file not found at {args.ppo_file}")
    except Exception as e:
        print(f"Error reading PPO file: {e}")

    # --- Load Dagger Data ---
    dagger_data_valid = False
    dagger_df = None
    try:
        if os.path.exists(args.dagger_file):
            dagger_df = pd.read_csv(args.dagger_file)
            # Note: Strip whitespace from column names
            dagger_df.columns = dagger_df.columns.str.strip()

            if "Iteration" in dagger_df.columns and "return_mean" in dagger_df.columns:
                dagger_data_valid = True
                print(f"Successfully loaded Dagger data from {args.dagger_file}")
            else:
                print(
                    f"Warning: Dagger file {args.dagger_file} missing required columns ('Iteration', 'return_mean'). Found: {dagger_df.columns.tolist()}"
                )
        else:
            print(f"Warning: Dagger file not found at {args.dagger_file}")
    except Exception as e:
        print(f"Error reading Dagger file: {e}")

    # --- Plot 1: PPO Graph ---
    if ppo_data_valid:
        plt.figure(figsize=(10, 6))
        plt.plot(
            ppo_df["total_timesteps"],
            ppo_df["ep_rew_mean"],
            label="PPO Mean Reward",
            color="blue",
            linewidth=2,
        )
        plt.title("PPO: Mean Reward vs Total Timesteps")
        plt.xlabel("Total Timesteps")
        plt.ylabel("Mean Reward")
        plt.yscale("symlog")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        output_path = os.path.join(args.output_dir, "ppo_training_curve.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Saved PPO graph to {output_path}")

    # --- Plot 2: Dagger Graph ---
    if dagger_data_valid:
        plt.figure(figsize=(10, 6))
        plt.plot(
            dagger_df["Iteration"],
            dagger_df["return_mean"],
            label="Dagger Mean Return",
            color="orange",
            linewidth=2,
        )
        plt.title("Dagger: Mean Return vs Total Timesteps")
        plt.xlabel("Total Timesteps")
        plt.ylabel("Mean Return")
        plt.yscale("symlog")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        output_path = os.path.join(args.output_dir, "dagger_training_curve.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Saved Dagger graph to {output_path}")

    # --- Plot 3: Combined Graph ---
    if ppo_data_valid and dagger_data_valid:
        plt.figure(figsize=(12, 7))
        plt.plot(
            ppo_df["total_timesteps"],
            ppo_df["ep_rew_mean"],
            label="PPO (ep_rew_mean)",
            color="blue",
            linewidth=2,
            alpha=0.8,
        )
        plt.plot(
            dagger_df["Iteration"],
            dagger_df["return_mean"],
            label="Dagger (return_mean)",
            color="orange",
            linewidth=2,
            alpha=0.8,
        )

        plt.title("Combined Training Metrics: PPO vs Dagger")
        plt.xlabel("Total Timesteps")
        plt.ylabel("Mean Reward / Return")
        plt.yscale("symlog")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()

        output_path = os.path.join(args.output_dir, "combined_training_curve.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Saved combined graph to {output_path}")

    if not ppo_data_valid and not dagger_data_valid:
        print("No valid data loaded. No graphs generated.")


if __name__ == "__main__":
    main()
