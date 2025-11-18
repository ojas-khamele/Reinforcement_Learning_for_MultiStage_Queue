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

# Loss column mapping per algorithm
LOSS_MAP = {
    "r2d2": ["Loss"],
    "ppo": ["ActorLoss", "CriticLoss"],
    "a2c": ["ActorLoss", "CriticLoss"],
    "dyna": ["TD_Error"]
}

# ============================================================
# LOAD LOG FILE
# ============================================================
def load_log(algorithm, seed):
    file_path = os.path.join(LOG_DIR, f"{algorithm}_training_log_{seed}.csv")
    return pd.read_csv(file_path)


# ============================================================
# MOVING AVERAGE
# ============================================================
def moving_average(data, window):
    return data.groupby(data.index // window).mean()


# ============================================================
# MODE 1: Plot loss across seeds (same algorithm)
# ============================================================
def plot_loss_across_seeds(algorithm, window=50):
    plt.figure(figsize=(10, 6))

    for loss_col in LOSS_MAP[algorithm]:
        for seed in SEEDS:
            df = load_log(algorithm, seed)
            avg_loss = moving_average(df[loss_col], window)
            plt.plot(avg_loss, label=f"{loss_col} - Seed {seed}")

    plt.title(f"Loss Comparison Across Seeds ({algorithm.upper()})")
    plt.xlabel(f"Training Steps (averaged every {window})")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ============================================================
# MODE 2: Plot loss for a single algorithm + single seed
# ============================================================
def plot_single_loss(algorithm, seed, window=50):
    plt.figure(figsize=(10, 6))

    df = load_log(algorithm, seed)
    for loss_col in LOSS_MAP[algorithm]:
        avg_loss = moving_average(df[loss_col], window)
        plt.plot(avg_loss, label=f"{loss_col}")

    plt.title(f"{algorithm.upper()} Loss Curve (Seed={seed})")
    plt.xlabel(f"Training Steps (averaged every {window})")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN LOOP
# ============================================================
while True:
    print("\n===================================")
    print(" RL Loss Visualization Tool")
    print("===================================\n")
    print("Choose plotting mode:")
    print("1 → Plot loss across SEEDS for an algorithm")
    print("2 → Plot loss for a specific ALGORITHM + SEED")
    print("q → Quit")
    
    choice = input("\nEnter choice (1, 2 or q): ").lower()

    if choice == "q":
        print("Exiting tool. Goodbye!\n")
        break

    if choice not in ["1", "2"]:
        print("Invalid choice! Try again.\n")
        continue

    window = int(input("Enter averaging window (e.g., 20, 50, 100): "))

    if choice == "1":
        algorithm = input("Enter algorithm name (a2c, dyna, ppo, r2d2): ").lower()
        print("\nPlotting loss across seeds...\n")
        plot_loss_across_seeds(algorithm, window)

    elif choice == "2":
        algorithm = input("Enter algorithm name (a2c, dyna, ppo, r2d2): ").lower()
        seed = int(input("Enter seed to plot (10, 20, 30, 40, 50): "))
        print("\nPlotting single loss curve...\n")
        plot_single_loss(algorithm, seed, window)
