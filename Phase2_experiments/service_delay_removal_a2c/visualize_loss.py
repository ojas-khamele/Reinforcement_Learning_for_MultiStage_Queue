import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# CONFIG
# ============================================================
LOG_DIR = "runs"

ALGORITHM = "a2c"
SEEDS = [10, 20, 30, 40, 50]

# Loss columns for A2C
LOSS_COLS = ["ActorLoss", "CriticLoss"]

# ============================================================
# LOAD LOG FILE
# ============================================================
def load_log(seed):
    file_path = os.path.join(LOG_DIR, f"{ALGORITHM}_training_log_{seed}.csv")
    return pd.read_csv(file_path)


# ============================================================
# MOVING AVERAGE
# ============================================================
def moving_average(data, window):
    return data.groupby(data.index // window).mean()


# ============================================================
# MODE 1: Plot loss across seeds
# ============================================================
def plot_loss_across_seeds(window=50):
    plt.figure(figsize=(10, 6))

    for loss_col in LOSS_COLS:
        for seed in SEEDS:
            df = load_log(seed)
            avg_loss = moving_average(df[loss_col], window)
            plt.plot(avg_loss, label=f"{loss_col} - Seed {seed}")

    plt.title("A2C Loss Comparison Across Seeds")
    plt.xlabel(f"Training Steps (averaged every {window})")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ============================================================
# MODE 2: Plot loss for a single seed
# ============================================================
def plot_single_loss(seed, window=50):
    plt.figure(figsize=(10, 6))

    df = load_log(seed)
    for loss_col in LOSS_COLS:
        avg_loss = moving_average(df[loss_col], window)
        plt.plot(avg_loss, label=f"{loss_col}")

    plt.title(f"A2C Loss Curve (Seed={seed})")
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
    print(" A2C Loss Visualization Tool")
    print("===================================\n")
    print("Choose plotting mode:")
    print("1 → Plot loss across SEEDS")
    print("2 → Plot loss for a specific SEED")
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
        print("\nPlotting A2C loss across seeds...\n")
        plot_loss_across_seeds(window)

    elif choice == "2":
        seed = int(input("Enter seed to plot (10, 20, 30, 40, 50): "))
        print("\nPlotting A2C loss curve...\n")
        plot_single_loss(seed, window)