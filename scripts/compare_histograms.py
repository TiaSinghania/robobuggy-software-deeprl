import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Compare episode length histograms from two monitor.csv files.")
    parser.add_argument("file1", type=str, help="Path to the first monitor.csv file")
    parser.add_argument("file2", type=str, help="Path to the second monitor.csv file")
    parser.add_argument("-k", "--k", type=int, required=True, help="Number of last rows to consider from each file")
    parser.add_argument("-o", "--output", type=str, default="histogram_comparison.png", help="Output image file path (default: histogram_comparison.png)")
    parser.add_argument("--name1", type=str, help="Label for the first file in the legend")
    parser.add_argument("--name2", type=str, help="Label for the second file in the legend")

    args = parser.parse_args()

    # Function to read and process monitor.csv
    def load_last_k_episodes(filepath, k):
        if not os.path.exists(filepath):
            print(f"Error: File not found at {filepath}")
            return None

        try:
            # Read csv, skipping the first line which is metadata
            df = pd.read_csv(filepath, skiprows=1)

            # Clean column names (remove whitespace)
            df.columns = df.columns.str.strip()

            if 'l' not in df.columns:
                print(f"Error: Column 'l' (episode length) not found in {filepath}. Found columns: {df.columns.tolist()}")
                return None

            return df.tail(k)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return None

    df1 = load_last_k_episodes(args.file1, args.k)
    df2 = load_last_k_episodes(args.file2, args.k)

    if df1 is None or df2 is None:
        print("Aborting due to file read errors.")
        return

    if len(df2) < args.k:
        print(f"Warning: File 2 had fewer than {args.k} rows (found {len(df2)}).")

    label1 = args.name1 if args.name1 else f'File 1 ({os.path.basename(args.file1)})'
    label2 = args.name2 if args.name2 else f'File 2 ({os.path.basename(args.file2)})'

    # Plot Episode Length Histogram
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(df1['l'], bins=50, alpha=0.5, label=label1, color='blue', edgecolor='black')
    plt.hist(df2['l'], bins=50, alpha=0.5, label=label2, color='orange', edgecolor='black')

    plt.title(f"Episode Length (Last {args.k} Episodes)")
    plt.xlabel("Episode Length (l)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)

    # Plot Rewards Histogram
    plt.subplot(1, 2, 2)
    # Check if 'r' column exists
    if 'r' in df1.columns and 'r' in df2.columns:
        # Create bins that are logarithmic for better visualization on symlog scale
        all_r = np.concatenate([df1['r'].dropna().values, df2['r'].dropna().values])

        if len(all_r) > 0:
            min_r, max_r = np.min(all_r), np.max(all_r)

            # Transform data to "log-space" (using log1p-like symlog transform)
            # y = sign(x) * log10(|x| + 1)
            def symlog_transform(x):
                return np.sign(x) * np.log10(np.abs(x) + 1)

            def symlog_inverse(y):
                return np.sign(y) * (10**np.abs(y) - 1)

            t_min, t_max = symlog_transform(min_r), symlog_transform(max_r)
            # Create linearly spaced bins in the transformed space
            t_bins = np.linspace(t_min, t_max, 100)
            # Transform back to data space
            bins = symlog_inverse(t_bins)
        else:
            bins = 50 # Fallback

        plt.hist(df1['r'], bins=bins, alpha=0.5, label=label1, color='blue', edgecolor='black')
        plt.hist(df2['r'], bins=bins, alpha=0.5, label=label2, color='orange', edgecolor='black')
        plt.title(f"Rewards (Last {args.k} Episodes)")
        plt.xlabel("Reward (r) - Log Scale X")
        plt.ylabel("Frequency")
        plt.xscale('symlog')
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.3)
    else:
        print("Warning: 'r' column not found in one of the files. Skipping rewards histogram.")

    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Histograms saved to {args.output}")
    plt.close()

if __name__ == "__main__":
    main()

