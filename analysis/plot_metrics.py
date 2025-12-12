import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Dict
import argparse

def load_metrics(log_dir: str) -> pd.DataFrame:
    metrics_path = os.path.join(log_dir, "metrics.jsonl")
    if not os.path.exists(metrics_path):
        print(f"No metrics file found at {metrics_path}")
        return pd.DataFrame()
    
    data = []
    with open(metrics_path, "r") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(data)

def calculate_derived_metrics(df: pd.DataFrame, epsilon: float = 0.1):
    if df.empty:
        return df
    
    # Calculate g - f
    df["g_minus_f"] = df["reward/compilable_g"] - df["reward/correctness_f"]
    
    # Calculate slope of g - f (using a window)
    window_size = 5
    df["slope_g_minus_f"] = df["g_minus_f"].rolling(window=window_size).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)
    
    # Calculate Cumulative Area Under Curve for g - f
    # Using trapezoidal rule cumulatively
    df["auc_g_minus_f"] = np.trapz(df["g_minus_f"], df["step"]) # This is total AUC, not cumulative.
    # For cumulative:
    df["auc_g_minus_f"] = 0.0
    if len(df) > 1:
        # Simple cumulative sum of (y * dx) assuming dx=1 step for simplicity, or use actual steps
        # Using simple cumsum for now as steps are usually 1
        df["auc_g_minus_f"] = df["g_minus_f"].cumsum()

    return df

def plot_metrics(df: pd.DataFrame, output_dir: str):
    if df.empty:
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Fundamental Metrics (f, g, Entropy)
    plt.figure(figsize=(12, 6))
    plt.plot(df["step"], df["reward/correctness_f"], label="Correctness (f)", color="green")
    plt.plot(df["step"], df["reward/compilable_g"], label="Compilable (g)", color="blue")
    plt.plot(df["step"], df["policy/entropy"], label="Entropy", color="orange", linestyle="--")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title("Training Metrics: f, g, and Entropy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "metrics_overview.png"))
    plt.close()

    # Plot 2: Reward Hacking Gap (g - f)
    plt.figure(figsize=(10, 5))
    plt.plot(df["step"], df["g_minus_f"], label="g - f (Hacking Gap)", color="red")
    plt.xlabel("Step")
    plt.ylabel("Gap")
    plt.title("Reward Hacking Gap (g - f)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "hacking_gap.png"))
    plt.close()

    # Plot 3: Derived Metrics (Slope and AUC)
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:purple'
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Slope of (g-f)', color=color)
    ax1.plot(df["step"], df["slope_g_minus_f"], color=color, label="Slope (g-f)")
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:brown'
    ax2.set_ylabel('Cumulative AUC (g-f)', color=color)  # we already handled the x-label with ax1
    ax2.plot(df["step"], df["auc_g_minus_f"], color=color, linestyle="--", label="Cumulative AUC")
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Derived Metrics: Slope and AUC of Hacking Gap")
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(os.path.join(output_dir, "derived_metrics.png"))
    plt.close()
    
    print(f"Plots saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Plot training metrics")
    parser.add_argument("--log_dir", type=str, default="/tmp/tinker-examples/rl-leetcode-qwen-4b", help="Directory containing metrics.jsonl")
    parser.add_argument("--output_dir", type=str, default="analysis_plots", help="Directory to save plots")
    args = parser.parse_args()

    print(f"Reading metrics from {args.log_dir}...")
    df = load_metrics(args.log_dir)
    
    if not df.empty:
        print(f"Found {len(df)} steps.")
        df = calculate_derived_metrics(df)
        plot_metrics(df, args.output_dir)
    else:
        print("No data found.")

if __name__ == "__main__":
    main()
