import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# CONFIG
# ============================================================
LOG_DIR = "logs"

ALGORITHMS = ["a2c", "dyna", "ppo", "r2d2"]
SEEDS = [10, 20, 30, 40, 50]

# Baseline reference values
BASE_MEAN = 37.34
BASE_STD = 6.83
BASE_MIN = 23.30
BASE_MAX = 48.12


# ============================================================
# HELPER FUNCTION TO LOAD LOG
# ============================================================
def load_log(algorithm, seed):
    file_path = os.path.join(LOG_DIR, f"{algorithm}_training_log_{seed}.csv")
    return pd.read_csv(file_path)


# ============================================================
# HELPER: COMPUTE ROLLING AVERAGE
# ============================================================
def moving_average(data, window):
    return data.groupby(data.index // window).mean()


# ============================================================
# PLOTTING BASELINE
# ============================================================
def plot_baseline(ax, x_points):
    ax.axhline(BASE_MEAN, color="black", linestyle="--", label="Baseline Mean")

    ax.fill_between(
        x_points,
        BASE_MEAN - BASE_STD,
        BASE_MEAN + BASE_STD,
        color="gray",
        alpha=0.2,
        label="Baseline ± STD"
    )

    ax.axhline(BASE_MIN, color="red", linestyle=":", label="Baseline Min")
    ax.axhline(BASE_MAX, color="green", linestyle=":", label="Baseline Max")


# ============================================================
# 1st WAY: Across Algorithms (fixed seed)
# ============================================================
def plot_across_algorithms(seed, window=50):
    plt.figure(figsize=(10, 6))

    for algo in ALGORITHMS:
        df = load_log(algo, seed)
        rewards = df["Reward"]
        avg_rewards = moving_average(rewards, window)
        plt.plot(avg_rewards, label=f"{algo.upper()}")

    x_points = np.arange(len(avg_rewards))
    plot_baseline(plt.gca(), x_points)

    plt.title(f"Reward Comparison Across Algorithms (Seed={seed})")
    plt.xlabel(f"Episode (averaged over {window})")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ============================================================
# 2nd WAY: Across Seeds (fixed algorithm)
# ============================================================
def plot_across_seeds(algorithm, window=50):
    plt.figure(figsize=(10, 6))

    for seed in SEEDS:
        df = load_log(algorithm, seed)
        rewards = df["Reward"]
        avg_rewards = moving_average(rewards, window)
        plt.plot(avg_rewards, label=f"Seed {seed}")

    x_points = np.arange(len(avg_rewards))
    plot_baseline(plt.gca(), x_points)

    plt.title(f"Reward Comparison Across Seeds ({algorithm.upper()})")
    plt.xlabel(f"Episode (averaged over {window})")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ============================================================
# 3rd WAY: Single Algorithm + Single Seed
# ============================================================
def plot_single_run(algorithm, seed, window=50):
    plt.figure(figsize=(10, 6))

    df = load_log(algorithm, seed)
    rewards = df["Reward"]
    avg_rewards = moving_average(rewards, window)

    plt.plot(avg_rewards, label=f"{algorithm.upper()} (Seed {seed})")

    x_points = np.arange(len(avg_rewards))
    plot_baseline(plt.gca(), x_points)

    plt.title(f"Reward Plot ({algorithm.upper()}, Seed={seed})")
    plt.xlabel(f"Episode (averaged over {window})")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ============================================================
# MENU LOOP (runs until 'q')
# ============================================================

while True:
    print("\n===================================")
    print(" RL Reward Visualization Tool")
    print("===================================")
    print("Choose plotting mode:")
    print("1 → Compare algorithms (fix seed)")
    print("2 → Compare seeds (fix algorithm)")
    print("3 → Plot single algorithm & single seed")
    print("q → Quit")

    mode = input("\nEnter choice (1, 2, 3, or q): ").strip().lower()

    if mode == 'q':
        print("Exiting... Goodbye!")
        break

    if mode not in ['1', '2', '3']:
        print("Invalid choice! Try again.")
        continue

    # Window input
    window = int(input("Enter averaging window (e.g., 20, 50, 100): "))

    if mode == '1':
        seed = int(input("Enter seed to plot (10, 20, 30, 40, 50): "))
        print("\nPlotting across algorithms...\n")
        plot_across_algorithms(seed, window)

    elif mode == '2':
        algorithm = input("Enter algorithm name (a2c, dyna, ppo, r2d2): ").lower()
        print("\nPlotting across seeds...\n")
        plot_across_seeds(algorithm, window)

    elif mode == '3':
        algorithm = input("Enter algorithm name (a2c, dyna, ppo, r2d2): ").lower()
        seed = int(input("Enter seed (10, 20, 30, 40, 50): "))
        print("\nPlotting single run...\n")
        plot_single_run(algorithm, seed, window)
